import habitat_sim
import magnum as mn

import cv2
import pandas as pd
# from tqdm import tqdm
import torch
import cv2
import math
import quaternion
import numpy as np
import torchvision.transforms as transforms
import clip
import os
import math
import json
import scipy
import matplotlib.pyplot as plt
import io
import time
from PIL import Image
from bevbuilder.utils.clip_mapping_utils import depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point
from bevbuilder.lseg.modules.models.lseg_net import LSegEncNet
from bevbuilder.lseg.additional_utils.models import resize_image, pad_image, crop_image
# from scipy.spatial.transform import Rotation as R
class CreateMap:
    def __init__(self, model_req_q=None, model_resp_q=None):
        # self.args = args
        # self.env = env
        # self.prompt_manager = prompt_manager
        self.actionlist = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        # self.SpatialKnowledgeGraph = nx.Graph() 
        self.Trajectory = [] 
        self.intersections = [] 
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.dead_end=0
        self.check_down_elevation=False
        self.num_backtrack=0
        self.num_reflect=0
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0
        self.semantic_bev = None
        self.color_bev = None        
        self.distance_bev = None
        self.habitat_sim = None
        self.semantic_inited = False
        self.model = None
        self.text_feats = None
        self.transform = None
        self.clip_feat_dim = None
        self.labels = None
        self.colors = None
        self.norm_mean = None
        self.norm_std = None
        # queues for model_server (None = load models in-process)
        self.model_req_q = model_req_q
        self.model_resp_q = model_resp_q
        self.init_tf_inv=None  
        self.color_top_down_height = None       
        # self.grid = None
        self.semantic_bev_ids = None
        self.obstacles = None
        # self.weight = None
        self.traj_points = []
        self.traj_idx_map = {}
        self.startX = 0
        self.startY = 0
        self.startR = 0
        # floor tracking: reference Z (world) and threshold (meters) to detect floor changes
        self.floor_ref_z = None
        self.floor_delta_thresh = 2.0
        self.profile_log_path = "bev_profile.log"

    def _profile_log(self, message):
        # Profiling/logging disabled per user request. Previously wrote to
        # `bev_profile.log` and printed messages; left as no-op to avoid
        # generating log files or stdout output.
        return

    def reset(self):
        # if self.SpatialKnowledgeGraph is not None:
        #     self.SpatialKnowledgeGraph.clear()
        
        self.Trajectory = []
        self.intersections = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        self.actionlist = []
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.check_down_elevation=False
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0
        camera_height = 1.5
        if self.semantic_inited:
            self.semantic_bev.fill(0)
            self.color_bev.fill(0)      
            self.distance_bev.fill(-1)
            self.color_top_down_height.fill(camera_height+1)
            # self.grid.fill(0)
            self.semantic_bev_ids.fill(-1) 
            self.obstacles.fill(1)
            # self.weight.fill(0)
        self.init_tf_inv= None
        self.traj_points = []
        self.traj_idx_map = {}
        self.startX = 0
        self.startY = 0
        self.startR = 0
        self.floor_ref_z = None

        if self.habitat_sim is not None:
            self.habitat_sim.close()
            self.habitat_sim = None   

    def drawNextNode(self, bev_save_path="bevbuilder/bev/map.png", cs=0.01, gs=3000):
        """Draw only next_node and agent->next dashed line on saved BEV images.

        Uses `bevbuilder/bev/node_location.txt` field `next_node_viewpointId` and
        the corresponding node entry in `other_nodes`.
        """

        def transform_point_to_new_frame(point_world, origin_pose):
            x0, y0, theta0 = origin_pose
            c, s = np.cos(theta0), np.sin(theta0)
            T_inv = np.array([
                [c, s, -c * x0 - s * y0],
                [-s, c, s * x0 - c * y0],
                [0, 0, 1],
            ])
            p_world_hom = np.array([point_world[0], point_world[1], 1])
            p_new_hom = T_inv @ p_world_hom
            return p_new_hom[:2]

        out_dir = os.path.dirname(bev_save_path) or "."
        node_path = os.path.join(out_dir, "node_location.txt")
        if not os.path.exists(node_path):
            node_path = os.path.join("bevbuilder", "bev", "node_location.txt")
        if not os.path.exists(node_path):
            return False

        try:
            with open(node_path, 'r') as f:
                node_json = json.load(f)
        except Exception:
            return False

        if not isinstance(node_json, dict):
            return False

        next_node_vp = node_json.get("next_node_viewpointId")
        other_nodes = node_json.get("other_nodes", [])
        if next_node_vp is None:
            return False

        target_item = None
        for item in other_nodes:
            if item.get("viewpointId") == next_node_vp:
                target_item = item
                break
        if target_item is None:
            return False

        center = gs // 2
        wx = float(target_item.get('x'))
        wy = float(target_item.get('y'))
        agent_z = -wy
        tmp_pos = pos2grid_id(gs, cs, wx - self.startX, agent_z - self.startY)
        new_pos = transform_point_to_new_frame(tmp_pos, (center, center, self.startR))
        newint_pos = np.round(new_pos).astype(int)
        max_offset = center - 1
        newint_pos = np.clip(newint_pos, -max_offset, max_offset)
        cand_x = int(center + newint_pos[0])
        cand_y = int(center - newint_pos[1])

        if len(self.traj_points) > 0:
            tx, ty = int(self.traj_points[-1][0]), int(self.traj_points[-1][1])
        else:
            tx, ty = gs // 2, gs // 2

        def _add_dashed_circle(img_bgr, center_xy, radius_m=2.5, cs_val=0.01):
            if img_bgr is None or img_bgr.size == 0:
                return img_bgr

            out = img_bgr.copy()
            h, w = out.shape[:2]
            cx, cy = int(center_xy[0]), int(center_xy[1])
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                return out

            radius_px = int(round(float(radius_m) / max(float(cs_val), 1e-6)))
            radius_px = max(1, radius_px)
            dash_deg = 12
            gap_deg = 8
            thickness = 2
            circle_color = (0, 0, 255)

            ang = 0
            while ang < 360:
                start_ang = ang
                end_ang = min(ang + dash_deg, 360)
                cv2.ellipse(out, (cx, cy), (radius_px, radius_px), 0, start_ang, end_ang, circle_color, thickness, lineType=cv2.LINE_AA)
                ang += (dash_deg + gap_deg)

            return out

        def _draw_on_bgr(img_bgr):
            if img_bgr is None:
                return img_bgr
            h, w = img_bgr.shape[:2]
            if not (0 <= cand_x < w and 0 <= cand_y < h):
                return img_bgr

            out = img_bgr.copy()
            cand_color = (0, 255, 0)
            cand_radius = 8
            cv2.circle(out, (cand_x, cand_y), cand_radius, cand_color, -1, lineType=cv2.LINE_AA)

            dn = target_item.get('display_number')
            if dn is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text = str(dn)
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                tx_text = int(cand_x - tw / 2)
                ty_text = int(cand_y + th / 2)
                b, g, r = cand_color
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = (0, 0, 0) if lum > 180 else (255, 255, 255)
                cv2.putText(out, text, (tx_text, ty_text), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

            full_len = math.hypot(cand_x - tx, cand_y - ty)
            if full_len > 1.0:
                vx = (cand_x - tx) / full_len
                vy = (cand_y - ty) / full_len
                traj_radius = 10
                start_offset = min(traj_radius, max(0.0, full_len - 0.5))
                sx0 = tx + vx * start_offset
                sy0 = ty + vy * start_offset
                rem_len = full_len - start_offset
                draw_len = max(0.0, rem_len - cand_radius)
                cur = 0.0
                dash_len = 12
                gap_len = 8
                while cur < draw_len:
                    s = cur
                    e = min(cur + dash_len, draw_len)
                    sx = int(round(sx0 + vx * s))
                    sy = int(round(sy0 + vy * s))
                    ex = int(round(sx0 + vx * e))
                    ey = int(round(sy0 + vy * e))
                    cv2.line(out, (sx, sy), (ex, ey), (0, 200, 0), 2, lineType=cv2.LINE_AA)
                    cur += dash_len + gap_len
            return out

        # map_color
        color_path = bev_save_path.replace('.png', '_color.png')
        color_img = cv2.imread(color_path)
        if color_img is not None:
            color_img = _draw_on_bgr(color_img)
            cv2.imwrite(color_path, color_img)

        # map_semantic_legend: draw on semantic top part only, keep legend area
        sem_path = bev_save_path.replace('.png', '_semantic_legend.png')
        sem_img = cv2.imread(sem_path)
        if sem_img is not None:
            h, w = sem_img.shape[:2]
            sem_h = min(h, gs)
            sem_top = sem_img[:sem_h, :, :].copy()
            sem_legend = sem_img[sem_h:, :, :].copy() if h > sem_h else None
            sem_top = _draw_on_bgr(sem_top)
            if sem_legend is not None and sem_legend.size > 0:
                sem_img_new = np.vstack([sem_top, sem_legend])
            else:
                sem_img_new = sem_top
            cv2.imwrite(sem_path, sem_img_new)

        # regenerate map_zoom_color based on updated map_color (same logic as before)
        try:
            color_img_new = cv2.imread(color_path)
            if color_img_new is not None:
                zoom_side = int(gs // 2)
                half = zoom_side // 2
                x0 = tx - half
                y0 = ty - half
                x1 = x0 + zoom_side
                y1 = y0 + zoom_side

                h, w = color_img_new.shape[:2]
                sx = max(0, x0)
                sy = max(0, y0)
                ex = min(w, x1)
                ey = min(h, y1)

                zoom_crop = color_img_new[sy:ey, sx:ex].copy()

                pad_top = max(0, -y0)
                pad_left = max(0, -x0)
                pad_bottom = max(0, y1 - h)
                pad_right = max(0, x1 - w)
                if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                    zoom_crop = cv2.copyMakeBorder(
                        zoom_crop,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

            red_zoom_x = int(round(tx - x0))
            red_zoom_y = int(round(ty - y0))
            zoom_crop = _add_dashed_circle(zoom_crop, (red_zoom_x, red_zoom_y), radius_m=2.5, cs_val=cs)
            # zoom_crop = _add_scale_bar(zoom_crop, scale_m=2.5, cs_val=cs, gs_val=gs)
            cv2.imwrite(bev_save_path.replace('.png', '_zoom_color.png'), zoom_crop)
        except Exception:
            pass

        # regenerate map_zoom_semantic_legend based on updated map_semantic_legend (same logic as before)
        try:
            sem_img_new = cv2.imread(sem_path)
            if sem_img_new is not None:
                h, w = sem_img_new.shape[:2]
                sem_h = min(h, gs)
                sem_top = sem_img_new[:sem_h, :, :].copy()
                sem_legend = sem_img_new[sem_h:, :, :].copy() if h > sem_h else None

                top_sem_h, top_sem_w = sem_top.shape[:2]
                zoom_side = int(min(top_sem_h, top_sem_w) // 2)
                zoom_side = max(1, zoom_side)
                half = zoom_side // 2

                x0 = tx - half
                y0 = ty - half
                # keep fixed zoom size by shifting crop window into valid bounds
                max_x0 = max(0, top_sem_w - zoom_side)
                max_y0 = max(0, top_sem_h - zoom_side)
                x0 = max(0, min(x0, max_x0))
                y0 = max(0, min(y0, max_y0))
                x1 = x0 + zoom_side
                y1 = y0 + zoom_side

                zoom_sem = sem_top[y0:y1, x0:x1].copy()

                if sem_legend is not None and sem_legend.size > 0:
                    legend_zoom = sem_legend
                    if legend_zoom.shape[1] != zoom_side:
                        new_legend_h = max(1, int(round(legend_zoom.shape[0] * (zoom_side / float(legend_zoom.shape[1])))))
                        legend_zoom = cv2.resize(legend_zoom, (zoom_side, new_legend_h), interpolation=cv2.INTER_AREA)
                    zoom_sem_legend = np.vstack([zoom_sem, legend_zoom])
                else:
                    zoom_sem_legend = zoom_sem

                red_zoom_x = int(round(tx - x0))
                red_zoom_y = int(round(ty - y0))
                zoom_sem_legend = _add_dashed_circle(zoom_sem_legend, (red_zoom_x, red_zoom_y), radius_m=2.5, cs_val=cs)
                # zoom_sem_legend = _add_scale_bar(zoom_sem_legend, scale_m=2.5, cs_val=cs, gs_val=gs)
                cv2.imwrite(bev_save_path.replace('.png', '_zoom_semantic_legend.png'), zoom_sem_legend)
        except Exception:
            pass

        return True

    
    def get_lseg_feat(self, model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        # vis_image = image.copy()
        image = transform(image).unsqueeze(0).cuda()
        # img = image[0].permute(1,2,0)
        # img = img * 0.5 + 0.5

        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(crop_size * stride_rate)

        long_size = base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height


        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= crop_size:
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                outputs, logits = model(pad_img, labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, norm_mean,
                                    norm_std, crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, norm_mean,
                                                norm_std, crop_size)
                    with torch.no_grad():
                        output, logits = model(pad_crop_img, labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        # outputs = outputs.numpy() # B, D, H, W
        # predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        # pred = predicts[0]

        outputs_np = outputs.numpy()  
        outputs_np = outputs_np.copy()
        return outputs_np
        # return outputs


    # def create_lseg_map_multiview(self, sim, ob, color_top_down, semantic_top_down, cs=0.01, gs=5000, depth_sample_rate=1, bev_save_path="bev/map.png"):
    def create_lseg_map_multiview(self, scan_id, viewpoint_id, current_heading, current_elevation, LX, LY, LZ, cs=0.01, gs=3000, depth_sample_rate=1, bev_save_path="bevbuilder/bev/map.png", save_stop_maps=False, is_reverie=False):
        request_t0 = time.perf_counter()
        request_tag = f"scan={scan_id} vp={viewpoint_id} heading={current_heading:.4f} elev={current_elevation:.4f}"
        self._profile_log(f"[BEV][START] {request_tag}")

        def draw_agent_trajectory(bev_image, trajectory, 
                                agent_color=(0,0,255), traj_color=(255,0,0), 
                                point_color=(0,255,0),  
                                arrow=False):

            img = bev_image.copy()

            for i in range(1, len(trajectory)):
                pt1 = (trajectory[i-1][0], trajectory[i-1][1])
                pt2 = (trajectory[i][0], trajectory[i][1])
                # cv2.arrowedLine(img, pt1, pt2, traj_color, 2, tipLength=0.2)
                cv2.line(img, pt1, pt2, traj_color, 2)

     
            # draw viewpoint circles and indices; last point is agent (red)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            for seq_idx, pt in enumerate(trajectory):
                x, y = int(pt[0]), int(pt[1])
                is_agent = (seq_idx == len(trajectory) - 1)
                circ_color = agent_color if is_agent else traj_color
                # determine display index: reuse original assigned index if present
                display_idx = None
                try:
                    display_idx = self.traj_idx_map.get((x, y), None)
                except Exception:
                    display_idx = None
                if display_idx is None:
                    display_idx = seq_idx
                # draw circle
                cv2.circle(img, (x, y), 10, circ_color, -1)

                # determine text color (white on dark circles, black on light)
                b, g, r = circ_color
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = (0, 0, 0) if lum > 180 else (255, 255, 255)

                # draw index text centered
                text = str(display_idx)
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                tx = int(x - tw / 2)
                ty = int(y + th / 2)
                cv2.putText(img, text, (tx, ty), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

            return img


        def init_semantic():
            # read labels and colors from MP3D's mpcat40.tsv
            df = pd.read_csv("mpcat40.tsv", sep='\t')
            df = df[(df['mpcat40index'] >= 1) & (df['mpcat40index'] <= 40)]
            labels = df['mpcat40'].tolist()
            colors = [[int(h[1:][i:i+2], 16) for i in (0, 2, 4)] for h in df['hex']]
            # print("labels:", labels)
            # print("colors:", colors)
            # loading models
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(device)
            clip_version = "ViT-B/32"
            clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                            'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
            print("Loading CLIP model...")
            clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
            clip_model.to(device).eval()
            lang_token = clip.tokenize(labels)
            lang_token = lang_token.to(device)
            with torch.no_grad():
                text_feats = clip_model.encode_text(lang_token)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # text_feats = text_feats.cpu().numpy()
            text_feats = text_feats.detach().cpu().numpy().copy().astype(np.float32)
            model = LSegEncNet(labels, arch_option=0,
                                block_depth=0,
                                activation='lrelu',
                                crop_size=crop_size)
            model_state_dict = model.state_dict()
            pretrained_state_dict = torch.load("lseg/checkpoints/demo_e200.ckpt",weights_only=False)
            pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
            model_state_dict.update(pretrained_state_dict)
            model.load_state_dict(pretrained_state_dict)

            model.eval()
            model = model.cuda()

            norm_mean= [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            # padding = [0.0] * 3
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            return model, text_feats, transform, clip_feat_dim, labels, colors, norm_mean, norm_std
        

        def transform_point_to_new_frame(point_world, origin_pose):
            x0, y0, theta0 = origin_pose
            c, s = np.cos(theta0), np.sin(theta0)

            T_inv = np.array([
                [ c,  s, -c*x0 - s*y0],
                [-s,  c,  s*x0 - c*y0],
                [ 0,  0, 1]
            ])

            p_world_hom = np.array([point_world[0], point_world[1], 1])
            p_new_hom = T_inv @ p_world_hom

            return p_new_hom[:2]

        def _add_scale_bar(img_bgr, scale_m=2.5, cs_val=0.01, gs_val=3000, margin=16):
            if img_bgr is None or img_bgr.size == 0:
                return img_bgr

            out = img_bgr.copy()
            h, w = out.shape[:2]
            if h <= 0 or w <= 0:
                return out

            # Compute accurate bar length on zoom image using cs, gs and zoom factor.
            # full_map_width_m = gs * cs; zoom_factor = gs / w.
            # visible_width_m = full_map_width_m / zoom_factor.
            # px_per_m = w / visible_width_m.
            try:
                zoom_factor = float(gs_val) / float(w)
                visible_width_m = (float(gs_val) * float(cs_val)) / zoom_factor
                px_per_m = float(w) / max(visible_width_m, 1e-6)
                bar_len_px = int(round(float(scale_m) * px_per_m))
            except Exception:
                bar_len_px = int(round(float(scale_m) / max(float(cs_val), 1e-6)))

            max_len = max(8, w - 2 * margin - 40)
            bar_len_px = max(8, min(bar_len_px, max_len))

            y = margin + 8
            x2 = w - margin
            x1 = max(margin, x2 - bar_len_px)

            color = (255, 255, 255)
            thickness = 2

            cv2.line(out, (x1, y), (x2, y), color, thickness, lineType=cv2.LINE_AA)
            tick_h = 6
            cv2.line(out, (x1, y - tick_h), (x1, y + tick_h), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(out, (x2, y - tick_h), (x2, y + tick_h), color, thickness, lineType=cv2.LINE_AA)

            text = f"{scale_m} meters"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_thick = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thick)
            tx = x2 - tw
            ty = min(h - margin, y + th + 12)
            cv2.putText(out, text, (tx, ty), font, font_scale, color, text_thick, lineType=cv2.LINE_AA)

            return out

        def _add_dashed_circle(img_bgr, center_xy, radius_m=2.5, cs_val=0.01):
            if img_bgr is None or img_bgr.size == 0:
                return img_bgr

            out = img_bgr.copy()
            h, w = out.shape[:2]
            cx, cy = int(center_xy[0]), int(center_xy[1])
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                return out

            radius_px = int(round(float(radius_m) / max(float(cs_val), 1e-6)))
            radius_px = max(1, radius_px)
            dash_deg = 12
            gap_deg = 8
            thickness = 2
            circle_color = (0, 0, 255)

            ang = 0
            while ang < 360:
                start_ang = ang
                end_ang = min(ang + dash_deg, 360)
                cv2.ellipse(out, (cx, cy), (radius_px, radius_px), 0, start_ang, end_ang, circle_color, thickness, lineType=cv2.LINE_AA)
                ang += (dash_deg + gap_deg)

            return out



        

        if self.habitat_sim is None:
            sim_cfg = habitat_sim.SimulatorConfiguration()
            # scan_id = ob['scan']
            # sim_cfg.scene_id = "mp3d/VLzqgDo317F/VLzqgDo317F.glb"
            sim_cfg.scene_id = f"bevbuilder/mp3d/{scan_id}/{scan_id}.glb"
            # sim_cfg.frustum_culling = False 
            # sim_cfg.gpu_device_id = 0   

            sensor_specs = []
            camera_height = 1.5
            rgb = habitat_sim.CameraSensorSpec()
            rgb.uuid = "rgb"
            rgb.sensor_type = habitat_sim.SensorType.COLOR
            rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            rgb.resolution = [480, 640]
            rgb.position = mn.Vector3(0, camera_height, 0)
            rgb.orientation = mn.Vector3(0, 0, 0)
            # rgb.vfov = 60

            sensor_specs.append(rgb)

            depth = habitat_sim.CameraSensorSpec()
            depth.uuid = "depth"
            depth.sensor_type = habitat_sim.SensorType.DEPTH
            depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth.resolution = [480, 640]
            depth.position = mn.Vector3(0, camera_height, 0)
            depth.orientation = mn.Vector3(0, 0, 0)
            # depth.vfov = 60
            sensor_specs.append(depth)

            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = sensor_specs
            # gs=3000
            sim_init_t0 = time.perf_counter()
            self._profile_log(f"[BEV][SIM_INIT][BEGIN] {request_tag} scene={sim_cfg.scene_id}")
            self.habitat_sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
            obs = self.habitat_sim.reset()
            self._profile_log(f"[BEV][SIM_INIT][END] {request_tag} dt={time.perf_counter() - sim_init_t0:.3f}s")
            if self.semantic_inited==False:
                self.color_bev = np.zeros((gs, gs, 3), dtype=np.uint8)
                self.semantic_bev = np.zeros((gs, gs, 3), dtype=np.uint8)


        # viewpoint_id = ob['viewpoint']
        # scan = ob['scan']
        color_top_down=self.color_bev
        semantic_top_down=self.semantic_bev
        # print("scan:", scan_id)
        # print("viewpoint_id:", viewpoint_id)
 
        # new_state = self.env.env.sims[0].getState()[0]
        # current_position = np.array([new_state.location.x, new_state.location.y, new_state.location.z])
        # current_heading = new_state.heading
        # print("current_heading:", current_heading)
        # current_elevation = new_state.elevation
        current_rotation = quaternion.from_euler_angles([current_elevation, current_heading, 0])  # [X, Y, Z]
        # r = R.from_euler('xyz', [current_elevation, current_heading, 0])
        # current_rotation = np.roll(r.as_quat(), 1)  # Convert to quaternion and roll to match habitat_sim's format
        # print("current_rotation:", current_rotation)


        agent_state = habitat_sim.AgentState()
  
        camera_height = 1.5 # MP3D camera height
        # print("mp3d current location:", new_state.location.x, new_state.location.y, new_state.location.z)
        agent_state.position = np.array(
            [LX, LZ-camera_height, -LY],
            # [new_state.location.x, new_state.location.z-camera_height, -new_state.location.y],
            dtype=np.float32
        )
       
        if len(self.traj_points) == 0:  
            self.startX = agent_state.position[0]
            self.startY = agent_state.position[2]
            self.startR = current_heading
        # Detect floor change by comparing provided LZ (world Z) to stored reference.
        # If the agent has moved to a different floor (vertical change > threshold),
        # reset BEV canvases and trajectory bookkeeping so BEVs from different floors
        # do not overlap.
        try:
            current_world_z = float(LZ)
        except Exception:
            current_world_z = None

        if current_world_z is not None:
            if self.floor_ref_z is None:
                self.floor_ref_z = current_world_z
            else:
                if abs(current_world_z - self.floor_ref_z) > self.floor_delta_thresh:
                    print(f"[INFO] Floor change detected: {self.floor_ref_z} -> {current_world_z}. Resetting BEV and trajectory.")
                    # Reinitialize BEV arrays (safe even if semantic not yet inited)
                    try:
                        self.color_bev = np.zeros((gs, gs, 3), dtype=np.uint8)
                        self.semantic_bev = np.zeros((gs, gs, 3), dtype=np.uint8)
                        self.distance_bev = -np.ones((gs, gs), dtype=np.float32)
                        self.color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
                        self.semantic_bev_ids = np.full((gs, gs), fill_value=-1, dtype=np.int16)
                        self.obstacles = np.ones((gs, gs), dtype=np.uint8)
                    except Exception:
                        # Fallback: ensure attributes exist
                        self.semantic_bev = self.semantic_bev if self.semantic_bev is not None else np.zeros((gs, gs, 3), dtype=np.uint8)
                        self.color_bev = self.color_bev if self.color_bev is not None else np.zeros((gs, gs, 3), dtype=np.uint8)
                    # Reset trajectory bookkeeping
                    self.traj_points = []
                    self.traj_idx_map = {}
                    # set new start reference on this floor
                    self.startX = agent_state.position[0]
                    self.startY = agent_state.position[2]
                    self.startR = current_heading
                    # update floor ref
                    self.floor_ref_z = current_world_z
          
        # print("startX:", self.startX, "startY:", self.startY, "startR:", self.startR)
       #old code
        # tmp_pos = pos2grid_id(gs, cs, agent_state.position[0]-self.startX, agent_state.position[2]-self.startY)
        # # print("tmp_pos:", tmp_pos)
        
        # new_pos = transform_point_to_new_frame(tmp_pos, (2500,2500,self.startR))
        # # print("new_pos:", new_pos)
        # newint_pos = np.round(new_pos).astype(int)
        # # print("newint_pos:", newint_pos)
        # bev_pos = (newint_pos[0] + 2500, 5000- (newint_pos[1] + 2500))
        # print("bev_pos:", bev_pos)
        # self.traj_points.append(bev_pos)
        #new code
        # compute grid pos relative to start (unchanged)
        tmp_pos = pos2grid_id(gs, cs, agent_state.position[0] - self.startX,
                            agent_state.position[2] - self.startY)

        # transform into the local BEV frame
        # use dynamic center instead of hard-coded 2500
        center = gs // 2
        new_pos = transform_point_to_new_frame(tmp_pos, (center, center, self.startR))
        newint_pos = np.round(new_pos).astype(int)

        # clamp offsets so points never fall outside the canvas
        max_offset = center - 1
        newint_pos = np.clip(newint_pos, -max_offset, max_offset)

        # map to image pixel coords: x increases to the right, y increases downward
        bev_x = int(center + newint_pos[0])
        bev_y = int(center - newint_pos[1])  # invert Y because BEV local y is up, image y is down

        # only append valid points
        if 0 <= bev_x < gs and 0 <= bev_y < gs:
            pos = (bev_x, bev_y)
            # if this exact viewpoint was seen before, keep its original index
            if pos in self.traj_idx_map:
                # append the existing position (retain index mapping)
                self.traj_points.append(pos)
            else:
                # assign next index and append
                next_idx = len(self.traj_idx_map)
                self.traj_idx_map[pos] = next_idx
                self.traj_points.append(pos)
        else:
            print(f"[DEBUG] skipped out-of-range traj point: {(bev_x, bev_y)}")



        agent_state.rotation = current_rotation
        # print("POSITION:", agent_state.position, type(agent_state.position))
        # print("ROTATION:", agent_state.rotation, type(agent_state.rotation))

        # return False
        self.habitat_sim.agents[0].set_state(agent_state)    

        # state = sim.get_agent(0).get_state()
        # camera_pos= state.sensor_states["rgb"].position
        # print("camera_pos:", camera_pos)
        # position = state.position

        # cs=0.01
        # gs=1000
        # depth_sample_rate=1
        # bev_save_path="map.png"

        # mask_version = 1 # 0, 1

        crop_size = 480 
        base_size = 520 

        if self.semantic_inited==False:
            semantic_init_t0 = time.perf_counter()
            self._profile_log(f"[BEV][SEMANTIC_INIT][BEGIN] {request_tag}")
            if self.model_req_q is not None:
                # model_server holds CLIP+LSeg; fetch pre-computed state from it
                self.model_req_q.put({"type": "init_info"})
                info = self.model_resp_q.get()
                if not info.get("ok"):
                    raise RuntimeError(f"model_server init_info failed: {info.get('err')}")
                self.text_feats = info["text_feats"]
                self.labels = info["labels"]
                self.colors = info["colors"]
                self.clip_feat_dim = info["clip_feat_dim"]
                self.norm_mean = info["norm_mean"]
                self.norm_std = info["norm_std"]
                self.model = None    # inference runs in model_server, not locally
                self.transform = None
            else:
                self.model, self.text_feats, self.transform, self.clip_feat_dim, self.labels, self.colors, self.norm_mean, self.norm_std = init_semantic()
            self.distance_bev = -np.ones((gs, gs), dtype=np.float32)
            self.color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)       
            # self.grid = np.zeros((gs, gs, self.clip_feat_dim), dtype=np.float32)
            self.semantic_bev_ids = np.full((gs, gs), fill_value=-1, dtype=np.int16)
            self.obstacles = np.ones((gs, gs), dtype=np.uint8)
            # self.weight = np.zeros((gs, gs), dtype=float)
            self.semantic_inited = True
            self._profile_log(f"[BEV][SEMANTIC_INIT][END] {request_tag} dt={time.perf_counter() - semantic_init_t0:.3f}s labels={len(self.labels)} feat_dim={self.clip_feat_dim}")
        distance_bev = self.distance_bev
        color_top_down_height=self.color_top_down_height       
        # grid=self.grid
        semantic_bev_ids=self.semantic_bev_ids
        obstacles=self.obstacles
        # weight=self.weight

        # semantic_pixel_bev = np.full((gs, gs), fill_value=-1, dtype=np.int32)

        yaw_list = list(range(0, 360, 90))
        tf_list = []

        # data_iter = zip(rgb_list, depth_list, semantic_list, pose_list)
        # pbar = tqdm(total=len(rgb_list))
        # for data_sample in data_iter:
        
        # for yaw in tqdm(yaw_list, desc="Processing Directions"):
        for yaw in yaw_list:
            yaw_t0 = time.perf_counter()
            self._profile_log(f"[BEV][YAW][BEGIN] {request_tag} yaw={yaw}")
            # rgb_path, depth_path, semantic_path, pose_path = data_sample
            # heading_rad = math.radians(yaw)
            # sim.newEpisode([scan_id], [viewpoint_id], [heading_rad], [0])
            # new_state = sim.getState()[0]
            # rgb = np.array(new_state.rgb, copy=False)
            # depth = np.squeeze(np.array(new_state.depth, copy=False)) * 0.00025  # MP3D simulator: 16-bit depth (0.25mm)
            
            # print(f"Processing yaw: {yaw} degrees")
            obs = self.habitat_sim.get_sensor_observations()
            rgb = np.array(obs["rgb"], dtype=np.uint8, copy=True)
            depth = np.array(obs["depth"], dtype=np.float32, copy=True)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            # cv2.imwrite(f"rgb_{yaw}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(f"depth_{yaw}.png", (depth * 255).astype("uint8"))

            state = self.habitat_sim.get_agent(0).get_state()
            camera_pos= state.sensor_states["rgb"].position
            # print("camera_pos:", camera_pos)
            # position = state.position
            position=np.array(state.position).copy()
            # pos = np.array(position)
            pos = np.array(position, dtype=np.float32, copy=True)

            rotation = state.rotation
            # print("rotation:", rotation, type(rotation))

            # rot = np.array(rotation.to_matrix())
            rot = quaternion.as_rotation_matrix(rotation)
            rot = np.array(rot, dtype=np.float32, copy=True)

            # r1=R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])  # habitat_sim uses [w, x, y, z]
            # rot = r1.as_matrix() 
            # print("position:", position)
            # print("rotation:", rot)
            
            delta_rot = quaternion.from_rotation_vector([0, math.radians(90), 0])
            # r2 = R.from_rotvec([0, math.radians(90), 0])
            # delta_rot=np.roll(r2.as_quat(), 1)  # Convert to quaternion and roll to match habitat_sim's format
            # delta_rot = R.from_rotvec([0, math.radians(90), 0])

            new_rotation = delta_rot * rotation
            state.rotation = new_rotation
            # sim.get_agent(0).set_state(state)
           

            # pos = np.array([position.x, position.y, position.z])

            # rgb_path = os.path.join("bev", f"rgb.png")
            # cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # depth_path = os.path.join("bev", f"depth.png")
            # depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth_vis = np.uint8(depth_norm)
            # cv2.imwrite(depth_path, depth_vis)
            
            # pix_feats = self.get_lseg_feat(model, rgb, label_list, transform)

            # bgr = cv2.imread(rgb_path)
            # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # read pose
            # pos, rot = load_pose(pose_path)  # z backward, y upward, x to the right
            # rot_ro_cam = np.eye(3)
            rot_ro_cam = np.eye(3, dtype=np.float32)
            rot_ro_cam[1, 1] = -1
            rot_ro_cam[2, 2] = -1

            # R_yaw = np.array([
            #     [np.cos(heading_rad), -np.sin(heading_rad), 0],
            #     [np.sin(heading_rad), np.cos(heading_rad), 0],
            #     [0, 0, 1]
            # ])
            # rot = R_yaw
            rot = rot @ rot_ro_cam  # z backward, y upward, x to the right

            # rot = rot @ rot_ro_cam
            pos[1] += camera_height  # set the camera height to 1.5m

            # pose = np.eye(4)
            pose = np.eye(4, dtype=np.float32) 
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)

            tf_list.append(pose)
            if len(tf_list) == 1:
                if(self.init_tf_inv is None):
                    self.init_tf_inv = np.linalg.inv(tf_list[0]).astype(np.float32, copy=True)
                    # self.init_tf_inv = np.linalg.inv(tf_list[0])
            
            tf = (self.init_tf_inv @ pose).astype(np.float32, copy=True)
            # tf = self.init_tf_inv @ pose
            # print("rot:\n", rot)
            # print("pose:\n", pose)
            # print("tf:\n", tf)

            # read depth
            # depth = load_depth(depth_path)

            # read semantic
            # semantic = load_semantic(semantic_path)
            # semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)
            # print("STEP 1: b4 get lsegs")

            lseg_t0 = time.perf_counter()
            if self.model_req_q is not None:
                self.model_req_q.put({"type": "infer", "rgb": rgb})
                _resp = self.model_resp_q.get()
                if not _resp.get("ok"):
                    raise RuntimeError(f"model_server infer failed: {_resp.get('err')}")
                pix_feats = _resp["pix_feats"]
            else:
                pix_feats = self.get_lseg_feat(self.model, rgb, self.labels, self.transform, crop_size, base_size, self.norm_mean, self.norm_std)
            # pix_feats = self.get_lseg_feat(model, rgb, label_list, transform)
            # print("STEP 2: after get lsegs")
            # pix_feats = np.array(pix_feats, copy=True)
            pix_feats = np.array(pix_feats, dtype=np.float32, copy=True)
            self._profile_log(f"[BEV][YAW][LSEG] {request_tag} yaw={yaw} dt={time.perf_counter() - lseg_t0:.3f}s feat_shape={tuple(pix_feats.shape)}")

            # transform all points to the global frame
            # pc, mask = depth2pc(depth, fov=30)
            depthproj_t0 = time.perf_counter()
            pc, mask = depth2pc(depth)
            # print("after depth2pc")
            pc   = np.array(pc,   dtype=np.float32, copy=True)  
            mask = np.array(mask, dtype=bool, copy=True) 
            # pc = np.array(pc, copy=True)                
            # mask = np.array(mask, copy=True)
            # pc, mask = depth2pc_real_world(depth, intrinsic_matrix)  

            shuffle_mask = np.arange(pc.shape[1])
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::depth_sample_rate]
            mask = mask[shuffle_mask]
            pc = pc[:, shuffle_mask]
            pc = pc[:, mask]
            pc_global = transform_pc(pc, tf)
            pc_global = np.array(pc_global, copy=True)
            point_count = int(pc.shape[1]) if pc.ndim == 2 else 0
            self._profile_log(f"[BEV][YAW][DEPTH2PC] {request_tag} yaw={yaw} dt={time.perf_counter() - depthproj_t0:.3f}s points={point_count}")
            # print("pix_feats shape:", pix_feats.shape)
            # assert len(pix_feats.shape) == 4 and pix_feats.shape[2] > 0 and pix_feats.shape[3] > 0
            # print(f"pix_feats type: {type(pix_feats)}")
            # print(f"pix_feats flags: {pix_feats.flags}")
            # print(f"pix_feats dtype: {pix_feats.dtype}")
            # print("pix_feats memory:", pix_feats.flags['OWNDATA'])  # 是否拥有数据
            # print("pix_feats contiguous:", pix_feats.flags['C_CONTIGUOUS'])
            rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
            # print("after get rgb cam mat")
            feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])
            # print("aa")

            # project all point cloud onto the ground

            project_t0 = time.perf_counter()
            updated_color = 0
            updated_semantic = 0
            obstacle_hits = 0
            skipped_oob = 0
            skipped_bad_norm = 0

            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x, y = pos2grid_id(gs, cs, p[0], p[2])
              
                
                # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
                if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                    x < 0 or y < 0 or p_local[1] < -0.5:
                    skipped_oob += 1
                    continue
                # print(f"points.shape = {p_local.shape}, mat.shape = {rgb_cam_mat.shape}")
                # print("points max abs:", np.max(np.abs(p_local)))
                # print("mat max abs:", np.max(np.abs(rgb_cam_mat)))

                # assert p_local is not None, "输入点云不能为 None"
                # assert isinstance(p_local, np.ndarray), "输入必须是 NumPy 数组"
                # assert np.isfinite(p_local).all(), "p_local contains non-finite values"
                # assert np.isfinite(feat_cam_mat).all(), "feat_cam_mat contains non-finite values"
                # assert np.isfinite(rgb_cam_mat).all(), "rgb_cam_mat contains non-finite values"
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                # print("bb")

                rgb_h, rgb_w = rgb.shape[:2]
                if rgb_px < 0 or rgb_px >= rgb_w or rgb_py < 0 or rgb_py >= rgb_h:
                    continue
                rgb_v = rgb[rgb_py, rgb_px, :]
                # semantic_v = semantic[rgb_py, rgb_px]
                # if semantic_v == 40:
                    # semantic_v = -1
                higher_point=False
                # when the projected location is already assigned a color value before, overwrite if the current point has larger height
                if p_local[1] < color_top_down_height[y, x]:
                    higher_point = True
                    color_top_down[y, x] = rgb_v
                    color_top_down_height[y, x] = p_local[1]
                    updated_color += 1
                    # gt[y, x] = semantic_v

                # select the highest and the closest semantic if multiple points are projected to the same grid cell
               

                px, py, pz = project_point(feat_cam_mat, p_local)
                # print("cc")

                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    # feat = pix_feats[0, :, py, px]
                    feat = np.array(pix_feats[0, :, py, px], dtype=np.float32, copy=True)
                    
                    norm = np.linalg.norm(feat)
                    if not (norm > 1e-4 and np.isfinite(norm)):
                        skipped_bad_norm += 1
                        continue
                    feat = feat / norm

                    # self.text_feats: (40, D), feat: (D,)
                    sim_scores = feat @ self.text_feats.T     # (40,)
                    cls_id = int(np.argmax(sim_scores))       # [0, num_labels)

                    if higher_point==True: #higher 
                        semantic_bev_ids[y, x] = cls_id
                        distance_bev[y, x] = p_local[2]
                        updated_semantic += 1

                    if p_local[1] == color_top_down_height[y, x]: #same height
                        if distance_bev[y, x] < 0 or p_local[2] < distance_bev[y, x]:
                            semantic_bev_ids[y, x] = cls_id
                            distance_bev[y, x] = p_local[2]
                            updated_semantic += 1
                    
               
                    # if higher_point==True: #higher
                    #     grid[y, x] = feat
                    #     distance_bev[y, x] = p_local[2]

                    # if p_local[1] == color_top_down_height[y, x]: #same height
                    #     # if np.any(feat != grid[y,x]): #new feat
                    #     if not np.allclose(feat, grid[y,x]):             
                    #         if p_local[2]< distance_bev[y, x]: #closer,so more accurate semantic
                    #             grid[y, x] = feat
                    #             distance_bev[y, x] = p_local[2]

                    # weight[y, x] += 1

                # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
                if p_local[1] > camera_height:
                    continue
                obstacles[y, x] = 0
                obstacle_hits += 1
            # print("dd")
            self.habitat_sim.get_agent(0).set_state(state) # change the agent state to the next heading
            self._profile_log(
                f"[BEV][YAW][PROJECT] {request_tag} yaw={yaw} dt={time.perf_counter() - project_t0:.3f}s "
                f"points={point_count} color_updates={updated_color} semantic_updates={updated_semantic} "
                f"obstacles={obstacle_hits} skipped_oob={skipped_oob} skipped_bad_norm={skipped_bad_norm}"
            )
            self._profile_log(f"[BEV][YAW][END] {request_tag} yaw={yaw} total_dt={time.perf_counter() - yaw_t0:.3f}s")


        # try:

        #     print("Starting semantic classification (vectorized)...")

        #     mask = weight > 0
        #     valid_feats = grid[mask]   # shape: (N, D)

        #     norms = np.linalg.norm(valid_feats, axis=1, keepdims=True)
        #     valid_feats = np.where(norms > 1e-4, valid_feats / norms, 0)
        #     # valid_mask = (norms > 1e-5) & np.isfinite(norms)  
        #     # valid_feats = np.where(valid_mask, valid_feats / norms, 0)  

        #     # valid_feats: (N, D), text_feats.T: (D, 40)
        #     sim_scores = valid_feats @ self.text_feats.T  # (N, 40)

        #     max_classes = np.argmax(sim_scores, axis=1)  # (N,)

        #     semantic_pixel_bev[mask] = max_classes

          
        # except Exception as e:
        #     print(f"Segmentation fault likely caused by: {e}")
        #     import traceback
        #     traceback.print_exc()    

        
        
        # draw trajectory on color BEV and save
        img_result = draw_agent_trajectory(color_top_down[:, :, ::-1], self.traj_points)
        cv2.imwrite(bev_save_path.replace(".png", "_color.png"), img_result)

        # save zoomed color map centered at red node (agent), with half-side crop
        try:
            if len(self.traj_points) > 0:
                cx, cy = int(self.traj_points[-1][0]), int(self.traj_points[-1][1])
            else:
                cx, cy = gs // 2, gs // 2

            zoom_side = int(gs // 2)
            half = zoom_side // 2
            x0 = cx - half
            y0 = cy - half
            x1 = x0 + zoom_side
            y1 = y0 + zoom_side

            h, w = img_result.shape[:2]
            sx = max(0, x0)
            sy = max(0, y0)
            ex = min(w, x1)
            ey = min(h, y1)

            zoom_crop = img_result[sy:ey, sx:ex].copy()

            pad_top = max(0, -y0)
            pad_left = max(0, -x0)
            pad_bottom = max(0, y1 - h)
            pad_right = max(0, x1 - w)
            if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                zoom_crop = cv2.copyMakeBorder(
                    zoom_crop,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            red_zoom_x = int(round(cx - x0))
            red_zoom_y = int(round(cy - y0))
            zoom_crop = _add_dashed_circle(zoom_crop, (red_zoom_x, red_zoom_y), radius_m=2.5, cs_val=cs)
            zoom_crop = _add_scale_bar(zoom_crop, scale_m=2.5, cs_val=cs, gs_val=gs)
            cv2.imwrite(bev_save_path.replace(".png", "_zoom_color.png"), zoom_crop)
        except Exception as e:
            print(f"Failed to build/save map zoom color image: {e}")

        # Also save a version with unvisited candidate nodes drawn in green
        try:
            # determine output directory and node location file
            out_dir = os.path.dirname(bev_save_path) or "."
            node_file_candidates = os.path.join(out_dir, "node_location.txt")
            if not os.path.exists(node_file_candidates):
                # fallback to expected bevbuilder/bev path
                node_file_candidates = os.path.join("bevbuilder", "bev", "node_location.txt")

            candidates_img = img_result.copy()
            if os.path.exists(node_file_candidates):
                try:
                    with open(node_file_candidates, 'r') as nf:
                        node_json = json.load(nf)
                except Exception:
                    node_json = None

                if node_json is not None:
                    # other_nodes are considered unvisited; they follow trajectory numbering
                    other_nodes = node_json.get('other_nodes', [])
                    # agent pixel center (if available)
                    agent_px = None
                    if len(self.traj_points) > 0:
                        agent_px = (int(self.traj_points[-1][0]), int(self.traj_points[-1][1]))

                    center = gs // 2
                    h, w = candidates_img.shape[:2]

                    # draw candidate nodes and record their pixel coordinates
                    cand_coord_map = {}
                    for item in other_nodes:
                        try:
                            # SKG/node positions are stored as (env_x, env_y, env_z).
                            # create_map's internal agent frame uses agent_x=env_x and agent_z=-env_y
                            # (see how agent_state.position is constructed earlier). Convert
                            # the SKG coordinates into that agent frame before mapping.
                            wx = float(item.get('x'))
                            wy = float(item.get('y'))
                            # convert to agent-frame z (negate env y)
                            agent_z = -wy
                            # compute grid pos relative to start like traj_points logic
                            tmp_pos = pos2grid_id(gs, cs, wx - self.startX, agent_z - self.startY)
                            # transform into the local BEV frame
                            new_pos = transform_point_to_new_frame(tmp_pos, (center, center, self.startR))
                            newint_pos = np.round(new_pos).astype(int)
                            max_offset = center - 1
                            newint_pos = np.clip(newint_pos, -max_offset, max_offset)
                            cand_x = int(center + newint_pos[0])
                            cand_y = int(center - newint_pos[1])

                            if 0 <= cand_x < gs and 0 <= cand_y < gs:
                                # draw green filled circle for unvisited node
                                cand_color = (0, 255, 0)
                                cand_radius = 8
                                cv2.circle(candidates_img, (cand_x, cand_y), cand_radius, cand_color, -1, lineType=cv2.LINE_AA)
                                # draw display number if present (centered on the circle)
                                dn = item.get('display_number')
                                if dn is not None:
                                    try:
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 0.4
                                        thickness = 1
                                        text = str(dn)
                                        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                                        tx = int(cand_x - tw / 2)
                                        ty = int(cand_y + th / 2)
                                        # choose black or white depending on marker brightness
                                        b, g, r = cand_color
                                        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                                        text_color = (0, 0, 0) if lum > 180 else (255, 255, 255)
                                        cv2.putText(candidates_img, text, (tx, ty), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
                                    except Exception:
                                        pass

                                # record pixel coord by display number for later mapping lines
                                try:
                                    if dn is not None:
                                        cand_coord_map[int(dn)] = (cand_x, cand_y, cand_radius)
                                except Exception:
                                    pass
                        except Exception:
                            continue
                # draw dashed lines only from the current agent (red) to candidate nodes
                try:
                    traj_radius = 10
                    # agent pixel center (if available)
                    agent_px = None
                    if len(self.traj_points) > 0:
                        agent_px = (int(self.traj_points[-1][0]), int(self.traj_points[-1][1]))

                    if agent_px is not None:
                        tx, ty = agent_px
                        # iterate all candidate coords and draw dashed line from agent to candidate
                        for entry in cand_coord_map.values():
                            try:
                                cx, cy, c_rad = entry
                                full_len = math.hypot(cx - tx, cy - ty)
                                if full_len <= 1.0:
                                    continue
                                vx = (cx - tx) / full_len
                                vy = (cy - ty) / full_len
                                start_offset = min(traj_radius, max(0.0, full_len - 0.5))
                                sx0 = tx + vx * start_offset
                                sy0 = ty + vy * start_offset
                                rem_len = full_len - start_offset
                                draw_len = max(0.0, rem_len - c_rad)
                                cur = 0.0
                                dash_len = 12
                                gap_len = 8
                                while cur < draw_len:
                                    s = cur
                                    e = min(cur + dash_len, draw_len)
                                    sx = int(round(sx0 + vx * s))
                                    sy = int(round(sy0 + vy * s))
                                    ex = int(round(sx0 + vx * e))
                                    ey = int(round(sy0 + vy * e))
                                    cv2.line(candidates_img, (sx, sy), (ex, ey), (0, 200, 0), 2, lineType=cv2.LINE_AA)
                                    cur += dash_len + gap_len
                            except Exception:
                                continue
                except Exception:
                    pass

                # save candidates overlay image
                candidates_path = os.path.join(out_dir, "map_candidates_color.png")
                cv2.imwrite(candidates_path, candidates_img)

                # Create a zoomed-in square centered on the agent (red node)
                try:
                    # determine agent center: last trajectory point if available, else image center
                    if len(self.traj_points) > 0:
                        cx, cy = int(self.traj_points[-1][0]), int(self.traj_points[-1][1])
                    else:
                        cx, cy = gs // 2, gs // 2

                    # new square side is half of previous (gs)
                    new_side = int(gs // 2)

                    half = new_side // 2
                    x0 = cx - half
                    y0 = cy - half
                    x1 = x0 + new_side
                    y1 = y0 + new_side

                    # clip to image bounds
                    h, w = candidates_img.shape[:2]
                    sx = max(0, x0)
                    sy = max(0, y0)
                    ex = min(w, x1)
                    ey = min(h, y1)

                    crop = candidates_img[sy:ey, sx:ex].copy()

                    # if crop is smaller than new_side (near edges), pad to maintain square size
                    ch, cw = crop.shape[:2]
                    pad_top = max(0, 0 - y0)
                    pad_left = max(0, 0 - x0)
                    pad_bottom = max(0, y1 - h)
                    pad_right = max(0, x1 - w)

                    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

                    # ensure final size exactly new_side x new_side
                    crop = cv2.resize(crop, (new_side, new_side), interpolation=cv2.INTER_LINEAR)

                    zoom_path = os.path.join(out_dir, "map_candidates_zoom_color.png")
                    cv2.imwrite(zoom_path, crop)
                except Exception as e:
                    print(f"Failed to build/save zoomed candidates image: {e}")
        except Exception as e:
            print(f"Failed to build/save candidates color map: {e}")

        # Also save a version with a dashed 3m-radius circle centered at the agent
        try:
            circle_img = img_result.copy()
            # determine center: last trajectory point if available, else image center
            if len(self.traj_points) > 0:
                cx, cy = int(self.traj_points[-1][0]), int(self.traj_points[-1][1])
            else:
                cx, cy = gs // 2, gs // 2

            # radius in pixels (3 meters default; 0.5m for reverie instruction)
            # radius_m = 0.5 if is_reverie else 3.0
            radius_m = 3.0

            radius_px = int(round(radius_m / cs))
            dash_deg = 12
            gap_deg = 8
            thickness = 2
            circle_color = (0, 0, 255)

            ang = 0
            while ang < 360:
                start_ang = ang
                end_ang = min(ang + dash_deg, 360)
                cv2.ellipse(circle_img, (cx, cy), (radius_px, radius_px), 0, start_ang, end_ang, circle_color, thickness)
                ang += (dash_deg + gap_deg)

            # not writing circle image to disk (user requested only map_color)
        except Exception as e:
            print(f"Failed to save map circle image: {e}")

        # build semantic_top_down image colors
        for i in range(len(self.labels)):
            semantic_top_down[semantic_bev_ids == i] = self.colors[i]

        # draw trajectory on semantic BEV (convert to BGR for drawing, then convert back)
        try:
            sem_bgr = semantic_top_down[:, :, ::-1].copy()
            sem_bgr_with_traj = draw_agent_trajectory(sem_bgr, self.traj_points)
            sem_with_traj = sem_bgr_with_traj[:, :, ::-1]
        except Exception:
            sem_with_traj = semantic_top_down.copy()

        # convert RGB -> BGR before saving so colors match OpenCV BGR convention used for color map
        try:
            sem_with_traj_bgr = sem_with_traj[:, :, ::-1]
        except Exception:
            sem_with_traj_bgr = sem_with_traj
        # not writing semantic BEV to disk (user requested only map_color)

        # obstacles: make 3-channel visualization and draw trajectory
        vis_obstacles = (obstacles * 255).astype(np.uint8)
        obs_color = np.stack([vis_obstacles] * 3, axis=-1)
        try:
            obs_with_traj = draw_agent_trajectory(obs_color, self.traj_points)
        except Exception:
            obs_with_traj = obs_color
        # not writing obstacles visualization to disk (user requested only map_color)

        # Helper to build a horizontal-wrapped legend image from a list of class ids
        def _build_legend_np(used_classes_list, gs_px, labels_list, colors_list):
            if len(used_classes_list) == 0:
                return np.zeros((1, gs_px, 3), dtype=np.uint8) + 255

            try:
                from PIL import ImageDraw, ImageFont
            except Exception:
                ImageDraw = None
                ImageFont = None

            swatch_w = 80
            padding_x = 16
            padding_y = 8
            entry_h = 48

            font = None
            font_size = 22
            try:
                if ImageFont is not None:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
            except Exception:
                font = None

            tmp_img = Image.new("RGB", (gs_px, entry_h), (255, 255, 255))
            tmp_draw = ImageDraw.Draw(tmp_img)

            item_metrics = []
            for cls in used_classes_list:
                label = str(labels_list[int(cls)]) if labels_list is not None else f"class_{int(cls)}"
                try:
                    text_w, text_h = tmp_draw.textsize(label, font=font)
                except Exception:
                    text_w, text_h = (100, entry_h - 2)
                item_w = swatch_w + padding_x + text_w + padding_x
                item_h = max(entry_h, text_h + 2 * padding_y)
                item_metrics.append((int(item_w), int(item_h), label, int(cls)))

            rows = []
            cur_row = []
            cur_row_w = 0
            for item_w, item_h, label, cls in item_metrics:
                if cur_row_w + item_w > gs_px and len(cur_row) > 0:
                    rows.append((cur_row, cur_row_w))
                    cur_row = []
                    cur_row_w = 0
                cur_row.append((item_w, item_h, label, cls))
                cur_row_w += item_w
            if len(cur_row) > 0:
                rows.append((cur_row, cur_row_w))

            legend_height = sum(max(it[1] for it in row[0]) for row in rows)
            legend_img = Image.new("RGB", (gs_px, max(legend_height, 1)), (255, 255, 255))
            draw = ImageDraw.Draw(legend_img)

            y_offset = 0
            for row_items, row_w in rows:
                x_offset = 0
                row_h = max(it[1] for it in row_items)
                for item_w, item_h, label, cls in row_items:
                    sw_x0 = x_offset
                    sw_y0 = y_offset + (row_h - entry_h) // 2
                    sw_x1 = sw_x0 + swatch_w
                    sw_y1 = sw_y0 + entry_h - 2
                    color = tuple(int(c) for c in colors_list[int(cls)])
                    draw.rectangle([sw_x0, sw_y0, sw_x1, sw_y1], fill=color)
                    text_x = sw_x1 + padding_x
                    text_y = y_offset + (row_h - entry_h) // 2 + padding_y
                    try:
                        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
                    except Exception:
                        draw.text((text_x, text_y), label, fill=(0, 0, 0))
                    x_offset += item_w
                y_offset += row_h

            return np.array(legend_img, dtype=np.uint8)

        try:
            used_classes = np.unique(semantic_bev_ids)
            used_classes = used_classes[used_classes >= 0]

            legend_np = _build_legend_np(list(map(int, used_classes)), gs, self.labels, self.colors)

            if legend_np.ndim == 2:
                legend_np = np.stack([legend_np] * 3, axis=-1)

            # use semantic image that includes the drawn trajectory
            try:
                top_sem = sem_with_traj
            except NameError:
                top_sem = semantic_top_down
            semantic_with_legend = np.vstack([top_sem, legend_np])
            try:
                semantic_with_legend_bgr = semantic_with_legend[:, :, ::-1]
            except Exception:
                semantic_with_legend_bgr = semantic_with_legend
            cv2.imwrite(bev_save_path.replace(".png", "_semantic_legend.png"), semantic_with_legend_bgr)

            # save zoomed semantic+legend map centered at red node (agent)
            try:
                if len(self.traj_points) > 0:
                    cx, cy = int(self.traj_points[-1][0]), int(self.traj_points[-1][1])
                else:
                    cx, cy = gs // 2, gs // 2

                # zoom around agent on semantic area, then append scaled legend
                top_sem_h, top_sem_w = top_sem.shape[:2]
                zoom_side = int(min(top_sem_h, top_sem_w) // 2)
                zoom_side = max(1, zoom_side)
                half = zoom_side // 2

                x0 = cx - half
                y0 = cy - half
                max_x0 = max(0, top_sem_w - zoom_side)
                max_y0 = max(0, top_sem_h - zoom_side)
                x0 = max(0, min(x0, max_x0))
                y0 = max(0, min(y0, max_y0))
                x1 = x0 + zoom_side
                y1 = y0 + zoom_side

                top_sem_bgr = top_sem[:, :, ::-1]
                zoom_sem = top_sem_bgr[y0:y1, x0:x1].copy()

                legend_bgr = legend_np[:, :, ::-1]
                if legend_bgr.shape[1] != zoom_side:
                    new_legend_h = max(1, int(round(legend_bgr.shape[0] * (zoom_side / float(legend_bgr.shape[1])))))
                    legend_bgr = cv2.resize(legend_bgr, (zoom_side, new_legend_h), interpolation=cv2.INTER_AREA)

                zoom_sem_legend = np.vstack([zoom_sem, legend_bgr])
                red_zoom_x = int(round(cx - x0))
                red_zoom_y = int(round(cy - y0))
                zoom_sem_legend = _add_dashed_circle(zoom_sem_legend, (red_zoom_x, red_zoom_y), radius_m=2.5, cs_val=cs)
                zoom_sem_legend = _add_scale_bar(zoom_sem_legend, scale_m=2.5, cs_val=cs, gs_val=gs)
                cv2.imwrite(bev_save_path.replace(".png", "_zoom_semantic_legend.png"), zoom_sem_legend)
            except Exception as e:
                print(f"Failed to build/save map zoom semantic legend: {e}")
        except Exception as e:
            print(f"Failed to build/save semantic legend: {e}")

        # If requested, create focused stop images with a 3m circular overlay centered at the agent
        if save_stop_maps:
            try:
                def _apply_focus_circle(img, center, radius_px, blur_strength=None, outside_dark=0.6):
                    h, w = img.shape[:2]
                    cx, cy = int(center[0]), int(center[1])

                    if blur_strength is None:
                        k = max(3, int(radius_px // 2) | 1)
                        max_k = (min(h, w) // 2) | 1
                        k = min(k, max_k)
                    else:
                        k = blur_strength
                        if k % 2 == 0:
                            k += 1

                    try:
                        blurred = cv2.GaussianBlur(img, (k, k), 0)
                    except Exception:
                        blurred = img.copy()

                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), int(radius_px), 255, -1)

                    out = blurred.copy()
                    out[mask == 255] = img[mask == 255]

                    if outside_dark is not None and outside_dark < 1.0:
                        outside_idx = (mask == 0)
                        out[outside_idx] = (out[outside_idx].astype(np.float32) * outside_dark).astype(np.uint8)

                    cv2.circle(out, (cx, cy), int(radius_px), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(out, (cx, cy), int(radius_px), (0, 0, 0), 1, lineType=cv2.LINE_AA)

                    return out

                # determine agent center in pixels (last trajectory point if available)
                if len(self.traj_points) > 0:
                    agent_px = self.traj_points[-1]
                    center_px = (int(agent_px[0]), int(agent_px[1]))
                else:
                    center_px = (gs // 2, gs // 2)

                # radius_m = 0.5 if is_reverie else 3.0
                radius_m = 3.0
                radius_px = int(round(radius_m / cs))

                # load color image we saved earlier (img_result in memory may be available as img_result)
                try:
                    color_img = img_result.copy()
                except Exception:
                    color_img = cv2.imread(bev_save_path.replace(".png", "_color.png"))

                focused_color = _apply_focus_circle(color_img, center_px, radius_px)

                # save into same folder as bev_save_path with fixed stop filenames
                out_dir = os.path.dirname(bev_save_path) or "."
                stop_color_path = os.path.join(out_dir, "map_stop_color.png")
                cv2.imwrite(stop_color_path, focused_color)

                # apply focus to semantic_with_legend (if available)
                try:
                    sem_img = semantic_with_legend.copy()
                except Exception:
                    sem_img = cv2.imread(bev_save_path.replace(".png", "_semantic_legend.png"))

                if sem_img is not None:
                    # ensure legend area (under the semantic map) is preserved
                    h, w = sem_img.shape[:2]
                    # semantic part expected to be gs x gs at the top
                    semantic_h = min(h, gs)
                    semantic_part = sem_img[0:semantic_h, :, :].copy()
                    legend_part = sem_img[semantic_h:, :, :].copy() if h > semantic_h else None

                    focused_sem_part = _apply_focus_circle(semantic_part, center_px, radius_px)

                    # build a legend containing only classes present inside the circle
                    yy = np.arange(semantic_part.shape[0])[:, None]
                    xx = np.arange(semantic_part.shape[1])[None, :]
                    cx, cy = center_px
                    # create mask for circle in semantic_part coords
                    mask_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius_px ** 2)
                    masked_ids = semantic_bev_ids.copy()
                    # ensure mask shapes match global indices (semantic_part is top portion)
                    masked_ids = masked_ids[0:semantic_part.shape[0], :]
                    present = np.unique(masked_ids[mask_circle])
                    present = present[present >= 0]

                    legend_np_stop = _build_legend_np(list(map(int, present)), gs, self.labels, self.colors)

                    if legend_part is not None and legend_part.size > 0:
                        combined = np.vstack([focused_sem_part, legend_np_stop])
                    else:
                        combined = focused_sem_part

                    stop_sem_path = os.path.join(out_dir, "map_stop_semantic_legend.png")
                    try:
                        combined_bgr = combined[:, :, ::-1]
                    except Exception:
                        combined_bgr = combined
                    cv2.imwrite(stop_sem_path, combined_bgr)
                    # Also create zoomed versions (half-side square) centered on agent
                    try:
                        # zoom side is half of gs
                        new_side = int(gs // 2)

                        # --- Zoom for color stop image ---
                        try:
                            # focused_color is gs x gs
                            cx, cy = center_px
                            half = new_side // 2
                            x0 = int(cx - half)
                            y0 = int(cy - half)
                            x1 = x0 + new_side
                            y1 = y0 + new_side

                            h, w = focused_color.shape[:2]
                            sx = max(0, x0)
                            sy = max(0, y0)
                            ex = min(w, x1)
                            ey = min(h, y1)

                            crop = focused_color[sy:ey, sx:ex].copy()

                            pad_top = max(0, 0 - y0)
                            pad_left = max(0, 0 - x0)
                            pad_bottom = max(0, y1 - h)
                            pad_right = max(0, x1 - w)

                            if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                                crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

                            crop = cv2.resize(crop, (new_side, new_side), interpolation=cv2.INTER_LINEAR)
                            stop_zoom_color_path = os.path.join(out_dir, "map_stop_zoom_color.png")
                            cv2.imwrite(stop_zoom_color_path, crop)
                        except Exception as e:
                            print(f"Failed to build/save stop zoom color: {e}")

                        # --- Zoom for semantic + legend ---
                        try:
                            # focused_sem_part contains gs x gs semantic (cropped & focused)
                            cx, cy = center_px
                            half = new_side // 2
                            x0 = int(cx - half)
                            y0 = int(cy - half)
                            x1 = x0 + new_side
                            y1 = y0 + new_side

                            sem_h, sem_w = focused_sem_part.shape[:2]
                            sx = max(0, x0)
                            sy = max(0, y0)
                            ex = min(sem_w, x1)
                            ey = min(sem_h, y1)

                            crop_sem = focused_sem_part[sy:ey, sx:ex].copy()

                            pad_top = max(0, 0 - y0)
                            pad_left = max(0, 0 - x0)
                            pad_bottom = max(0, y1 - sem_h)
                            pad_right = max(0, x1 - sem_w)

                            if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                                crop_sem = cv2.copyMakeBorder(crop_sem, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255,255,255))

                            zoom_sem_part = cv2.resize(crop_sem, (new_side, new_side), interpolation=cv2.INTER_LINEAR)

                            # If legend for the stop exists (legend_np_stop), scale it to new_side width and append below
                            try:
                                if 'legend_np_stop' in locals() and legend_np_stop is not None and legend_np_stop.size > 0:
                                    # scale legend width to new_side
                                    legend_h = legend_np_stop.shape[0]
                                    new_legend_h = max(1, int(round(legend_h * (new_side / float(gs)))))
                                    scaled_legend = cv2.resize(legend_np_stop, (new_side, new_legend_h), interpolation=cv2.INTER_AREA)
                                    # combine vertically (ensure both are RGB)
                                    combined_zoom = np.vstack([zoom_sem_part, scaled_legend])
                                else:
                                    combined_zoom = zoom_sem_part
                            except Exception:
                                combined_zoom = zoom_sem_part

                            try:
                                combined_zoom_bgr = combined_zoom[:, :, ::-1]
                            except Exception:
                                combined_zoom_bgr = combined_zoom
                            stop_zoom_sem_path = os.path.join(out_dir, "map_stop_zoom_semantic_legend.png")
                            cv2.imwrite(stop_zoom_sem_path, combined_zoom_bgr)
                        except Exception as e:
                            print(f"Failed to build/save stop zoom semantic legend: {e}")
                    except Exception as e:
                        print(f"Failed to build/save stop zoom images: {e}")
            except Exception as e:
                print(f"Failed to build/save stop-focused maps: {e}")

        self._profile_log(f"[BEV][END] {request_tag} total_dt={time.perf_counter() - request_t0:.3f}s traj_points={len(self.traj_points)}")
        return True