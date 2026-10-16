
import base64
import io
import os
import threading
import traceback

import torch
from flask import Flask, request, jsonify, Response
from PIL import Image, UnidentifiedImageError

from models import load_generators, translate, is_sketch


HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "generators.pth")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# Render's free instance has limited memory. Use CPU for predictable deployment.
device = torch.device("cpu")

G_AB = None
G_BA = None
SIZE = 256
model_error = None
model_lock = threading.Lock()


def load_models():
    """Load the models only when a conversion is requested."""
    global G_AB, G_BA, SIZE, model_error

    if G_AB is not None and G_BA is not None:
        return

    with model_lock:
        if G_AB is not None and G_BA is not None:
            return

        try:
            if not os.path.isfile(CHECKPOINT):
                raise FileNotFoundError(
                    f"Model checkpoint not found: {CHECKPOINT}"
                )

            app.logger.info("Loading CycleGAN models on CPU...")
            G_AB, G_BA, SIZE = load_generators(CHECKPOINT, device)
            G_AB.eval()
            G_BA.eval()
            model_error = None
            app.logger.info("CycleGAN models loaded successfully.")

        except Exception as exc:
            model_error = str(exc)
            app.logger.exception("Failed to load CycleGAN models.")
            raise


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CycleGAN Face ↔ Sketch</title>
<style>
body {
    font-family: system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    margin: 0;
    padding: 24px;
    text-align: center;
}
h1 { margin-bottom: 4px; }
.sub { color: #94a3b8; }
.card {
    background: #1e293b;
    border-radius: 14px;
    padding: 18px;
    max-width: 900px;
    margin: 16px auto;
}
button, select {
    background: #6366f1;
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px;
    font-size: 15px;
    cursor: pointer;
}
select { background: #334155; }
input[type=file] { display: none; }
.row {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
}
.box { flex: 1; min-width: 240px; }
.box img, video {
    width: 100%;
    max-width: 340px;
    border-radius: 10px;
    background: #0b1220;
    min-height: 120px;
    object-fit: contain;
}
#status {
    margin: 12px;
    color: #a5b4fc;
    min-height: 24px;
    overflow-wrap: anywhere;
}
</style>
</head>
<body>
<h1>Face ↔ Sketch (CycleGAN)</h1>
<p class="sub">
Upload a picture or use your camera. Convert between a face photo and a sketch.
</p>

<div class="card">
    <button onclick="document.getElementById('f').click()">
        Upload image
    </button>
    <input id="f" type="file" accept="image/*">

    <button onclick="startCam()">Use camera</button>
    <button id="snap" onclick="snap()" style="display:none">
        Capture &amp; convert
    </button>

    <select id="mode">
        <option value="auto">Auto-detect</option>
        <option value="photo">Input is a photo</option>
        <option value="sketch">Input is a sketch</option>
    </select>

    <div>
        <video id="v" autoplay playsinline style="display:none"></video>
        <canvas id="c" style="display:none"></canvas>
    </div>

    <div id="status">Ready</div>

    <div class="row">
        <div class="box">
            <h3 id="t1">Input</h3>
            <img id="in" alt="Input image">
        </div>
        <div class="box">
            <h3 id="t2">Output</h3>
            <img id="out" alt="Converted image">
        </div>
    </div>
</div>

<script>
const $ = id => document.getElementById(id);

async function send(blob) {
    if (!blob) {
        $('status').textContent = 'Could not capture an image.';
        return;
    }

    $('status').textContent = 'Converting image... Please wait.';
    $('out').removeAttribute('src');
    $('in').src = URL.createObjectURL(blob);

    const fd = new FormData();
    fd.append('image', blob, 'image.png');
    fd.append('mode', $('mode').value);

    try {
        const response = await fetch('/convert', {
            method: 'POST',
            body: fd
        });

        // Read the response as text first so empty or non-JSON
        // server responses do not cause an unexplained JSON error.
        const text = await response.text();
        let data;

        try {
            data = JSON.parse(text);
        } catch {
            throw new Error(
                'Server returned an invalid response (HTTP ' +
                response.status + '). Check Render logs.'
            );
        }

        if (!response.ok || data.error) {
            throw new Error(
                data.error || 'Request failed (HTTP ' + response.status + ').'
            );
        }

        $('out').src = 'data:image/png;base64,' + data.output;
        $('t1').textContent = 'Input (' + data.detected + ')';
        $('t2').textContent = 'Output (' + data.target + ')';
        $('status').textContent = 'Conversion complete!';

    } catch (error) {
        $('status').textContent = 'Error: ' + error.message;
        console.error(error);
    }
}

$('f').addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) send(file);
});

async function startCam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        $('v').srcObject = stream;
        $('v').style.display = 'inline-block';
        $('snap').style.display = 'inline-block';
    } catch (error) {
        $('status').textContent =
            'Camera unavailable. Allow camera permission and use HTTPS.';
    }
}

function snap() {
    const video = $('v');
    const canvas = $('c');

    if (!video.videoWidth || !video.videoHeight) {
        $('status').textContent = 'Camera is not ready yet.';
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => send(blob), 'image/png');
}
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    # This checks that Flask is responding, not that inference works.
    return jsonify({
        "status": "ok",
        "model_loaded": G_AB is not None and G_BA is not None,
        "device": str(device),
    }), 200


@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify(error="No image received. Please upload an image."), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify(error="Please select an image."), 400

    try:
        img = Image.open(uploaded_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="Could not read this image. Try a JPG or PNG."), 400

    try:
        mode = request.form.get("mode", "auto")

        if mode not in ("auto", "photo", "sketch"):
            return jsonify(error="Invalid conversion mode."), 400

        load_models()

        if mode == "auto":
            sketch = is_sketch(img)
        else:
            sketch = mode == "sketch"

        generator = G_BA if sketch else G_AB

        # translate() already uses no_grad in the supplied models.py.
        # inference_mode also makes inference-only intent explicit.
        with torch.inference_mode():
            output = translate(generator, img, SIZE)

        output = output.resize((256, 256), Image.Resampling.BICUBIC)

        return jsonify(
            output=image_to_base64(output),
            detected="sketch" if sketch else "real face",
            target="real face" if sketch else "sketch",
        )

    except Exception:
        app.logger.error("Image conversion failed:\n%s", traceback.format_exc())
        return jsonify(
            error=(
                "Image conversion failed on the server. "
                "Check the latest Render logs for details."
            )
        ), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Image is too large. Please upload an image under 10 MB."), 413


@app.errorhandler(500)
def internal_server_error(_error):
    app.logger.exception("Unhandled server error.")
    return jsonify(error="Internal server error. Check Render logs."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
