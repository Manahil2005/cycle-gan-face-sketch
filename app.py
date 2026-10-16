
import io
import base64
import os
import traceback

import torch
from flask import Flask, request, jsonify, Response
from PIL import Image

from models import load_generators, translate, is_sketch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "generators.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading CycleGAN models on {device}...", flush=True)

G_AB, G_BA, SIZE = load_generators(MODEL_PATH, device)

print("CycleGAN models loaded successfully.", flush=True)

app = Flask(__name__)

# Limit uploads to 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# --------------------------------------------------
# Image helpers
# --------------------------------------------------

def b64(img):
    """Convert a PIL image to a base64-encoded PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------
# Web interface
# --------------------------------------------------

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>CycleGAN Face ↔ Sketch</title>

<style>
* {
    box-sizing: border-box;
}

body {
    font-family: system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    margin: 0;
    padding: 24px;
    text-align: center;
}

h1 {
    margin: 0 0 8px;
}

p.sub {
    color: #94a3b8;
    margin-top: 0;
}

.card {
    background: #1e293b;
    border-radius: 14px;
    padding: 18px;
    max-width: 900px;
    margin: 16px auto;
}

button, select {
    background: #6366f1;
    color: #fff;
    border: 0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px;
    font-size: 15px;
    cursor: pointer;
}

button:disabled {
    opacity: 0.6;
    cursor: wait;
}

select {
    background: #334155;
}

input[type=file] {
    display: none;
}

.row {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
}

.box {
    flex: 1;
    min-width: 260px;
}

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
    min-height: 22px;
    overflow-wrap: anywhere;
}

#status.error {
    color: #fca5a5;
}

#status.success {
    color: #86efac;
}

.note {
    color: #94a3b8;
    font-size: 13px;
}
</style>
</head>

<body>

<h1>Face ↔ Sketch (CycleGAN)</h1>

<p class="sub">
Upload a picture or use your camera to convert a face into a sketch
or a sketch into a face.
</p>

<div class="card">

    <button id="uploadBtn"
        onclick="document.getElementById('f').click()">
        Upload image
    </button>

    <input id="f" type="file" accept="image/*">

    <button id="cameraBtn" onclick="startCam()">
        Use camera
    </button>

    <button id="snap" onclick="snap()" style="display:none">
        Capture &amp; convert
    </button>

    <button id="stopCam" onclick="stopCam()" style="display:none">
        Stop camera
    </button>

    <div style="margin-top: 12px;">
        <label for="mode">Conversion mode: </label>

        <select id="mode">
            <option value="auto">Auto-detect</option>
            <option value="photo">Input is a photo</option>
            <option value="sketch">Input is a sketch</option>
        </select>
    </div>

    <p class="note">
        Auto-detect may not classify every image correctly.
    </p>

    <div>
        <video id="v" autoplay playsinline
            style="display:none"></video>
        <canvas id="c" style="display:none"></canvas>
    </div>

    <div id="status" role="status" aria-live="polite">
        Ready. Upload an image to begin.
    </div>

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

let cameraStream = null;
let busy = false;

function setStatus(message, type = "") {
    $("status").textContent = message;
    $("status").className = type;
}

function setBusy(value) {
    busy = value;

    $("uploadBtn").disabled = value;
    $("cameraBtn").disabled = value;
    $("snap").disabled = value;
    $("mode").disabled = value;
}

async function send(blob) {
    if (!blob) {
        setStatus("Could not capture or read the image.", "error");
        return;
    }

    if (busy) return;

    setBusy(true);
    setStatus("Converting image. Please wait...");

    $("out").removeAttribute("src");
    $("t1").textContent = "Input";
    $("t2").textContent = "Output";

    const previousUrl = $("in").dataset.objectUrl;

    if (previousUrl) {
        URL.revokeObjectURL(previousUrl);
    }

    const inputUrl = URL.createObjectURL(blob);
    $("in").src = inputUrl;
    $("in").dataset.objectUrl = inputUrl;

    const fd = new FormData();
    fd.append("image", blob, "img.png");
    fd.append("mode", $("mode").value);

    try {
        const response = await fetch("/convert", {
            method: "POST",
            body: fd
        });

        // Read the response as text first. This lets us report
        // non-JSON responses instead of hiding the actual error.
        const responseText = await response.text();

        let data;

        try {
            data = JSON.parse(responseText);
        } catch (error) {
            throw new Error(
                "Server returned HTTP " + response.status +
                " without valid JSON. " +
                (responseText.slice(0, 200) || "The response was empty.")
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error || "Server error: HTTP " + response.status
            );
        }

        if (data.error) {
            throw new Error(data.error);
        }

        if (!data.output) {
            throw new Error("The server did not return a converted image.");
        }

        $("out").src = "data:image/png;base64," + data.output;

        $("t1").textContent = "Input (detected: " + data.detected + ")";
        $("t2").textContent = "Output (" + data.target + ")";

        setStatus("Conversion complete!", "success");

    } catch (error) {
        console.error("Image conversion failed:", error);

        setStatus(
            "Conversion failed: " + error.message,
            "error"
        );

    } finally {
        setBusy(false);
    }
}

$("f").addEventListener("change", event => {
    const file = event.target.files[0];

    if (file) {
        if (!file.type.startsWith("image/")) {
            setStatus("Please select a valid image file.", "error");
            return;
        }

        send(file);
    }

    // Allow the same file to be selected again.
    event.target.value = "";
});

async function startCam() {
    if (!navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia) {
        setStatus(
            "Camera is unavailable. Use HTTPS and a supported browser.",
            "error"
        );
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        $("v").srcObject = cameraStream;
        $("v").style.display = "inline-block";
        $("snap").style.display = "inline-block";
        $("stopCam").style.display = "inline-block";

        setStatus("Camera ready. Capture an image to convert it.");

    } catch (error) {
        console.error("Camera error:", error);

        setStatus(
            "Camera unavailable. Check browser permissions and HTTPS.",
            "error"
        );
    }
}

function stopCam() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    $("v").srcObject = null;
    $("v").style.display = "none";
    $("snap").style.display = "none";
    $("stopCam").style.display = "none";

    setStatus("Camera stopped.");
}

function snap() {
    if (busy) return;

    const video = $("v");
    const canvas = $("c");

    if (!video.videoWidth || !video.videoHeight) {
        setStatus("Camera is not ready yet. Please wait.", "error");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(
        video, 0, 0, canvas.width, canvas.height
    );

    canvas.toBlob(blob => {
        send(blob);
    }, "image/png");
}

window.addEventListener("beforeunload", () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
});
</script>

</body>
</html>
"""


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    """Basic endpoint to check whether Flask is responding."""
    return jsonify(status="ok", device=str(device))


@app.route("/convert", methods=["POST"])
def convert():
    print("Received image conversion request.", flush=True)

    if "image" not in request.files:
        return jsonify(error="No image received."), 400

    try:
        uploaded_file = request.files["image"]

        try:
            img = Image.open(uploaded_file.stream).convert("RGB")
        except Exception:
            return jsonify(error="Could not read the uploaded image."), 400

        mode = request.form.get("mode", "auto")

        if mode not in ("auto", "photo", "sketch"):
            return jsonify(error="Invalid conversion mode."), 400

        if mode == "auto":
            sketch = is_sketch(img)
        else:
            sketch = mode == "sketch"

        generator = G_BA if sketch else G_AB

        # Inference does not require gradients, reducing memory use.
        with torch.inference_mode():
            out = translate(generator, img, SIZE)

        out = out.resize((256, 256), Image.Resampling.BICUBIC)

        result = {
            "output": b64(out),
            "detected": "sketch" if sketch else "real face",
            "target": "real face" if sketch else "sketch"
        }

        print("Image conversion completed successfully.", flush=True)

        return jsonify(result)

    except Exception:
        traceback.print_exc()

        return jsonify(
            error="Image conversion failed. Please check the server logs."
        ), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        error="Image is too large. Please upload an image smaller than 10 MB."
    ), 413


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Unexpected server error")
    return jsonify(error="Internal server error."), 500


# --------------------------------------------------
# Start local development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
