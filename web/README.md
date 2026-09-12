# SpatialQwen-BEV V0 Replay Console

This is the first Web-first interface milestone for SpatialQwen-BEV. It is a self-contained React replay dashboard built from the repository's existing RGB observation, BEV map, semantic map, candidate map, node trajectory, and evaluation-log artifacts.

## Run locally

```bash
cd web
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. From the same network, use `http://10.79.128.145:4173`.

## What V0 includes

- A responsive navigation dashboard modeled on the supplied console reference.
- Color, semantic, and candidate BEV layers based on current repository artifacts.
- A functional replay timeline with previous, next, restart, play, and seek controls.
- Synchronized action, subtask, reasoning, confidence, and current-viewpoint panels.
- A real multi-turn task chat backed by the already-running local `qwen3-vl-8b` service.
- Map download and clear V0 boundaries for later session/backend integration.

## Local Qwen chat

The task console sends chat requests to `/api/qwen/chat/completions`. Vite proxies that request from the web server to the local OpenAI-compatible Qwen service, so `QWEN_API_KEY` is never sent to the browser.

The default target is `http://10.79.128.145:8001/v1` with model `qwen3-vl-8b`, matching the existing project API client. Override `QWEN_BASE_URL`, `QWEN_API_KEY`, or `VITE_QWEN_MODEL` in `web/.env` when the local deployment changes. The proxy is available through `npm run dev`.

## V0 boundary

V0 does not start Matterport, Habitat, Qwen, or the BEV worker. The replay map data remains a frontend fixture derived from repository artifacts and logs, while the task console can query an already-running local Qwen endpoint. V1 will replace the fixture with session-scoped FastAPI and WebSocket events.
