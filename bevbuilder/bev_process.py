import faulthandler
import multiprocessing as mp
import os
import time
import traceback

import zmq

from bevbuilder.utils.create_map import CreateMap
from bevbuilder.utils.model_server import model_server_loop


ENDPOINT = "tcp://*:5555"
WORKER_TIMEOUT_SEC = float(os.environ.get("BEV_WORKER_TIMEOUT_SEC", "1800"))


def _proc_log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    try:
        with open("bev_profile.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _enable_faulthandler(log_name):
    # Do not create per-process faulthandler log files anymore.
    # Enable faulthandler to write to stderr (default) but avoid opening
    # any dedicated log files or scheduling periodic dumps to files.
    try:
        faulthandler.enable()
    except Exception:
        pass


def _worker_loop(child_conn, model_req_q, model_resp_q):
    _enable_faulthandler(f"faulthandler_worker_{os.getpid()}.log")
    bevbuilder = CreateMap(model_req_q=model_req_q, model_resp_q=model_resp_q)
    while True:
        try:
            task = child_conn.recv()
        except EOFError:
            break

        task_type = task.get("type")
        if task_type == "quit":
            try:
                bevbuilder.reset()
            except Exception:
                pass
            child_conn.send({"ok": True})
            break

        if task_type == "draw_next_node":
            try:
                ok = bevbuilder.drawNextNode(
                    bev_save_path=task.get("bev_save_path", "bevbuilder/bev/map.png"),
                    cs=task.get("cs", 0.01),
                    gs=task.get("gs", 3000),
                )
                child_conn.send({"ok": bool(ok)})
            except Exception as e:
                tb = traceback.format_exc()
                child_conn.send({"ok": False, "err": str(e), "tb": tb})
            continue

        if task_type != "build":
            child_conn.send({"ok": False, "err": "unknown_worker_task"})
            continue

        scan = task.get("scan")
        viewpoint = task.get("viewpoint")
        heading = task.get("heading")
        elevation = task.get("elevation")
        LX = task.get("LX")
        LY = task.get("LY")
        LZ = task.get("LZ")
        is_reverie = task.get("is_reverie", False)

        req_t0 = time.perf_counter()
        try:
            _proc_log(f"[bev worker {os.getpid()}] start BEV build scan={scan} viewpoint={viewpoint}")
            bevbuilder.create_lseg_map_multiview(scan, viewpoint, heading, elevation, LX, LY, LZ, is_reverie=is_reverie)
            dt = time.perf_counter() - req_t0
            _proc_log(f"[bev worker {os.getpid()}] finish BEV build scan={scan} viewpoint={viewpoint} dt={dt:.3f}s")
            child_conn.send({"ok": True, "dt": dt})
        except Exception as e:
            tb = traceback.format_exc()
            _proc_log(f"[bev worker {os.getpid()}] build failed: {e}")
            child_conn.send({"ok": False, "err": str(e), "tb": tb})

    try:
        child_conn.close()
    except Exception:
        pass


class WorkerManager:
    def __init__(self):
        self.mp_ctx = mp.get_context("spawn")
        # model-server (singleton for entire lifetime of bev_process)
        self.model_req_q = self.mp_ctx.Queue()
        self.model_resp_q = self.mp_ctx.Queue()
        self.model_ready_event = self.mp_ctx.Event()
        self.model_server_proc = None
        # sim-worker (per-scan)
        self.worker_proc = None
        self.parent_conn = None
        self.current_scan = None
        self.worker_restart_count = 0
        self.worker_retry_count = 0
        self.graceful_stop_count = 0
        self.terminate_stop_count = 0
        self.kill_stop_count = 0

    def start_model_server(self):
        """Launch the model-server subprocess and wait until CLIP+LSeg are loaded."""
        proc = self.mp_ctx.Process(
            target=model_server_loop,
            args=(self.model_req_q, self.model_resp_q, self.model_ready_event),
            daemon=True,
        )
        proc.start()
        self.model_server_proc = proc
        _proc_log(f"[bev process] model_server started pid={proc.pid}; waiting for models to load …")
        if not self.model_ready_event.wait(timeout=180):
            raise RuntimeError("model_server did not become ready within 180 s")
        _proc_log(f"[bev process] model_server ready (pid={proc.pid})")

    def stop_model_server(self):
        """Send quit sentinel to model_server and wait for it to exit."""
        if self.model_server_proc is None:
            return
        try:
            self.model_req_q.put(None)  # sentinel
            self.model_server_proc.join(timeout=10)
        except Exception:
            pass
        if self.model_server_proc.is_alive():
            self.model_server_proc.terminate()
            self.model_server_proc.join(timeout=5)
        _proc_log(f"[bev process] model_server stopped pid={self.model_server_proc.pid}")
        self.model_server_proc = None

    def start_new_worker(self, reason, scan=None):
        prev_pid = self.worker_proc.pid if self.worker_proc is not None else None
        self.stop_worker()
        parent_conn, child_conn = self.mp_ctx.Pipe()
        worker_proc = self.mp_ctx.Process(
            target=_worker_loop,
            args=(child_conn, self.model_req_q, self.model_resp_q),
            daemon=True,
        )
        worker_proc.start()
        self.worker_proc = worker_proc
        self.parent_conn = parent_conn
        self.current_scan = scan
        self.worker_restart_count += 1
        _proc_log(
            f"[bev process] worker switched old_pid={prev_pid} -> new_pid={worker_proc.pid} "
            f"reason={reason} scan={scan} restart_count={self.worker_restart_count}"
        )

    def stop_worker(self):
        if self.worker_proc is None:
            return

        stop_mode = "none"
        try:
            if self.parent_conn is not None and self.worker_proc.is_alive():
                self.parent_conn.send({"type": "quit"})
                if self.parent_conn.poll(2.0):
                    _ = self.parent_conn.recv()
                    stop_mode = "graceful"
                    self.graceful_stop_count += 1
        except Exception:
            pass

        if self.worker_proc.is_alive():
            self.worker_proc.terminate()
            self.worker_proc.join(timeout=2.0)
            if not self.worker_proc.is_alive() and stop_mode == "none":
                stop_mode = "terminate"
                self.terminate_stop_count += 1
        if self.worker_proc.is_alive():
            self.worker_proc.kill()
            self.worker_proc.join(timeout=2.0)
            if stop_mode == "none":
                stop_mode = "kill"
                self.kill_stop_count += 1

        try:
            if self.parent_conn is not None:
                self.parent_conn.close()
        except Exception:
            pass

        _proc_log(
            f"[bev process] worker stopped pid={self.worker_proc.pid} mode={stop_mode} "
            f"stats(graceful={self.graceful_stop_count},terminate={self.terminate_stop_count},kill={self.kill_stop_count})"
        )
        self.worker_proc = None
        self.parent_conn = None
        self.current_scan = None

    def ensure_worker_for_msg(self, msg_type, scan):
        need_new = False
        reason = ""
        if self.worker_proc is None or not self.worker_proc.is_alive():
            need_new = True
            reason = "worker_missing"
        elif msg_type == "reset":
            need_new = True
            reason = "reset"
        elif self.current_scan is not None and scan != self.current_scan:
            need_new = True
            reason = "scan_switch"

        if need_new:
            self.start_new_worker(reason=reason, scan=scan)

    def build_once(self, msg):
        if self.worker_proc is None or self.parent_conn is None:
            return {"ok": False, "err": "worker_not_ready"}

        try:
            self.parent_conn.send({
                "type": "build",
                "scan": msg.get("scan"),
                "viewpoint": msg.get("viewpoint"),
                "heading": msg.get("heading"),
                "elevation": msg.get("elevation"),
                "LX": msg.get("LX"),
                "LY": msg.get("LY"),
                "LZ": msg.get("LZ"),
                "is_reverie": msg.get("is_reverie", False),
            })
        except Exception as e:
            return {"ok": False, "err": f"worker_send_failed: {e}"}

        if not self.parent_conn.poll(WORKER_TIMEOUT_SEC):
            return {"ok": False, "err": "worker_timeout"}

        try:
            return self.parent_conn.recv()
        except EOFError:
            return {"ok": False, "err": "worker_pipe_closed"}

    def build_with_retry(self, msg):
        scan = msg.get("scan")
        self.ensure_worker_for_msg(msg.get("type"), scan)
        result = self.build_once(msg)
        if result.get("ok"):
            return result

        self.worker_retry_count += 1
        _proc_log(
            f"[bev process] worker build failed once: {result.get('err')}; restarting worker and retrying "
            f"retry_count={self.worker_retry_count}"
        )
        self.start_new_worker(reason="retry", scan=scan)
        return self.build_once(msg)

    def draw_next_node_once(self, msg):
        if self.worker_proc is None or self.parent_conn is None:
            return {"ok": False, "err": "worker_not_ready"}

        try:
            self.parent_conn.send({
                "type": "draw_next_node",
                "bev_save_path": msg.get("bev_save_path", "bevbuilder/bev/map.png"),
                "cs": msg.get("cs", 0.01),
                "gs": msg.get("gs", 3000),
            })
        except Exception as e:
            return {"ok": False, "err": f"worker_send_failed: {e}"}

        if not self.parent_conn.poll(WORKER_TIMEOUT_SEC):
            return {"ok": False, "err": "worker_timeout"}

        try:
            return self.parent_conn.recv()
        except EOFError:
            return {"ok": False, "err": "worker_pipe_closed"}

    def draw_next_node_with_retry(self, msg):
        scan = msg.get("scan")
        self.ensure_worker_for_msg("step", scan)
        result = self.draw_next_node_once(msg)
        if result.get("ok"):
            return result

        self.worker_retry_count += 1
        _proc_log(
            f"[bev process] draw_next_node failed once: {result.get('err')}; restarting worker and retrying "
            f"retry_count={self.worker_retry_count}"
        )
        self.start_new_worker(reason="retry_draw_next_node", scan=scan)
        return self.draw_next_node_once(msg)


def main():
    _enable_faulthandler("faulthandler.log")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(ENDPOINT)
    _proc_log("[bev process] listening on port 5555")

    manager = WorkerManager()
    manager.start_model_server()
    while True:
        try:
            _proc_log("[bev process] waiting for request ...")
            msg = sock.recv_json()
            _proc_log(f"[bev process] recv msg: {msg}")
        except zmq.error.ZMQError as e:
            _proc_log(f"[bev process] recv error: {e}; continue")
            continue

        msg_type = msg.get("type")
        if msg_type == "quit":
            manager.stop_worker()
            manager.stop_model_server()
            try:
                sock.send_json({"ok": True})
            except Exception:
                pass
            _proc_log("[bev process] quit.")
            break

        if msg_type not in ("step", "reset"):
            try:
                sock.send_json({"ok": False, "err": "unknown_type"})
            except Exception:
                pass
            continue

        req_t0 = time.perf_counter()
        result = manager.build_with_retry(msg)
        if result.get("ok"):
            _proc_log(f"[bev process] send msg of bev maps ready dt={time.perf_counter() - req_t0:.3f}s")
        else:
            _proc_log(f"[bev process] build failed after retry: {result.get('err')}")

        try:
            sock.send_json({"ok": bool(result.get("ok", False)), "err": result.get("err")})
        except zmq.error.ZMQError as e:
            _proc_log(f"[bev process] reply failed (peer gone): {e}; drop and continue")
            continue

    manager.stop_worker()
    manager.stop_model_server()
    sock.close(0)
    ctx.term()


if __name__ == "__main__":
    main()


# Programmatic API for embedding the BEV manager inside the main process
def start_bev_manager():
    """Start and return a WorkerManager instance for in-process use.

    This preserves the per-scan worker subprocess model and the separate
    model-server process, but avoids using ZMQ. Call `stop_bev_manager`
    to cleanly shut it down.
    """
    _enable_faulthandler("faulthandler_inproc.log")
    manager = WorkerManager()
    manager.start_model_server()
    return manager


def stop_bev_manager(manager: WorkerManager):
    """Stop worker and model-server for a manager returned by
    `start_bev_manager()`.
    """
    try:
        manager.stop_worker()
    except Exception:
        pass
    try:
        manager.stop_model_server()
    except Exception:
        pass