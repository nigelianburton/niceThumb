import threading, time, uuid, traceback
from flask import Flask, request, jsonify, Response, stream_with_context
import logging
from werkzeug.serving import WSGIRequestHandler

# ==========================================================
# HTTP logging controls (added)
# ==========================================================
# Set QUIET_HTTP_LOG = True to suppress ALL request lines.
# Set QUIET_PROGRESS_ONLY = True to suppress only /progress/* polling lines.
# Both False => normal logging.
QUIET_HTTP_LOG = False
QUIET_PROGRESS_ONLY = True

# import debugpy
# debugpy.listen(("localhost", 5678))  # You can change the port if needed
# debugpy.wait_for_client()            # Pause until debugger attaches
# print("qtServer.py is waiting for debugger to attach...")



class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code='-', size='-'):
        # Suppress everything if full quiet
        if QUIET_HTTP_LOG:
            return
        # Suppress only progress polling noise
        if QUIET_PROGRESS_ONLY and getattr(self, "path", "").startswith("/progress/"):
            return
        super().log_request(code, size)

# ==========================================================
# Backend setup
# ==========================================================
# Legacy Qwen backend removed: all Qwen operations (t2i + edit/i2i via edit)
# are now handled exclusively by the enhanced QwenEdit1Backend.
# ==========================================================

try:
    from .qtdSDXL import SDXLBackend          # type: ignore
    from .qtdQEdit1 import QwenEdit1Backend   # type: ignore
except ImportError:
    from qtdSDXL import SDXLBackend           # type: ignore
    from qtdQEdit1 import QwenEdit1Backend    # type: ignore

app = Flask(__name__)

JOBS: dict[str, dict] = {}

# Only two backends now: SDXL and enhanced Qwen (qwen1).
BACKENDS = {
    "sdxl": SDXLBackend(),
    "qwen1": QwenEdit1Backend(),
}

MODEL_INDEX: dict[str, str] = {}  # modelId -> backendId


def rebuild_model_index():
    MODEL_INDEX.clear()
    for b in BACKENDS.values():
        try:
            for m in b.describe_models():
                mid = (m or {}).get("id")
                if mid:
                    MODEL_INDEX[mid] = b.id
        except Exception:
            print(f"[qtd][models][error] describe_models failed for backend={getattr(b,'id','?')}")
            traceback.print_exc()


rebuild_model_index()


@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    return jsonify({
        "status": "ok",
        "models": len(MODEL_INDEX),
        "jobs": len(JOBS),
        "time": time.time()
    })


@app.route("/models")
def models():
    out = []
    for b in BACKENDS.values():
        try:
            out.extend(b.describe_models())
        except Exception:
            print("[qtd][models][error] backend describe_models failed:")
            traceback.print_exc()
    return jsonify({"models": out})


def _loras_as_names(loras_raw):
    names = []
    for item in (loras_raw or []):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            n = item.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n)
    return names


@app.route("/loras")
def loras():
    model_id = request.args.get("modelId", "")
    backend_id = MODEL_INDEX.get(model_id, "")
    b = BACKENDS.get(backend_id)
    if not b:
        return jsonify({"loras": []})
    try:
        raw = b.list_loras(model_id)
    except Exception:
        print(f"[qtd][loras][error] list_loras failed for modelId={model_id}")
        traceback.print_exc()
        return jsonify({"loras": []})
    return jsonify({"loras": _loras_as_names(raw)})


@app.route("/jobs/submit", methods=["POST"])
def submit():
    payload = request.get_json(force=True)
    model_id = payload.get("modelId")
    op = payload.get("operation")
    inputs = payload.get("inputs") or {}
    loras = payload.get("loras")

    mask_active = payload.get("mask_active", None)  # <-- Accept optional bool

    if mask_active is not None:
        inputs["mask_active"] = bool(mask_active)  # <-- Pass to backend

    backend_id = MODEL_INDEX.get(model_id or "", "")
    b = BACKENDS.get(backend_id)

    # All Qwen (qwen1) requests already carry modelId "qwen1:image-edit".
    # No rerouting needed; if client still sends deprecated "qwen:image-edit"
    # the job will be rejected (intentional cleanup).
    if not (b and op):
        print(f"[qtd][job][reject] invalid modelId/operation modelId={model_id} op={op}")
        return jsonify({"error": "invalid modelId/operation"}), 400

    jid = uuid.uuid4().hex
    JOBS[jid] = {"status": "running", "percent": 0, "status_text": "Running"}

    def set_progress(p: int):
        JOBS[jid]["percent"] = int(max(0, min(100, p)))

    def set_result(img):
        JOBS[jid].update({
            "status": "done",
            "percent": 100,
            "result": img,
            "status_text": "Done"
        })
        print(f"[qtd][job][{jid}] done")

    def set_error(msg: str):
        print(f"[qtd][job][{jid}][error] {msg}")
        JOBS[jid].update({
            "status": "error",
            "error": str(msg),
            "status_text": "Error"
        })

    def set_status(txt: str):
        # Lightweight helper to update human readable phase text
        if isinstance(txt, str) and txt:
            JOBS[jid]["status_text"] = txt

    def _job_runner():
        print(f"[qtd][job][{jid}] start model={model_id} op={op}")
        try:
            # Unified pre-load step (non-fatal if backend handles lazy init)
            if hasattr(b, "ensure_pipeline"):
                try:
                    b.ensure_pipeline(model_id=model_id, operation=op,
                                      progress_cb=set_progress, status_cb=set_status)
                except Exception as e:
                    print(f"[qtd][job][{jid}][warn] ensure_pipeline failed (continuing to submit): {e}")
            b.submit(
                model_id=model_id,
                operation=op,
                inputs=inputs,
                loras=loras,
                set_progress=set_progress,
                set_result=set_result,
                set_error=set_error,
                set_status=set_status
            )
        except Exception as e:
            print(f"[qtd][job][{jid}][fatal] Unhandled exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            set_error(f"{type(e).__name__}: {e}")

    threading.Thread(target=_job_runner, name=f"job_{jid}", daemon=True).start()
    return jsonify({"status": "accepted", "job_id": jid}), 202


@app.route("/progress/<job_id>", methods=["GET"])
def api_progress(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    body = {k: j.get(k, "") for k in ("status", "percent", "status_text")}
    if j.get("status") == "done":
        body["image_data_url"] = j.get("result", "")
    if j.get("status") == "error":
        body["error"] = j.get("error", "")
    return jsonify(body)


@app.route("/events/<job_id>", methods=["GET"])
def api_events(job_id: str):
    def _gen():
        last = -1
        while True:
            j = JOBS.get(job_id)
            if not j:
                yield 'event: error\ndata: {"error":"unknown job"}\n\n'
                return
            pct = int(j.get("percent", 0))
            if pct != last:
                last = pct
                yield f'event: progress\ndata: {{"percent": {pct}}}\n\n'
            status = j.get("status")
            if status == "done":
                data_url = (j.get("result") or "").replace('"', '\\"')
                yield f'event: result\ndata: {{"image_data_url": "{data_url}"}}\n\n'
                return
            if status == "error":
                err = (j.get("error") or "").replace('"', '\\"')
                yield f'event: error\ndata: {{"error": "{err}"}}\n\n'
                return
            time.sleep(0.2)
    return Response(stream_with_context(_gen()), mimetype="text/event-stream")


if __name__ == "__main__":
    import os
    host = os.environ.get("QTD_HOST", "127.0.0.1")
    port = int(os.environ.get("QTD_PORT", "5015"))
    print(f"[qtd][startup] host={host} port={port}")
    rebuild_model_index()
    if QUIET_HTTP_LOG:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host=host, port=port, threaded=True, request_handler=QuietRequestHandler)