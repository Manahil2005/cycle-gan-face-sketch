import io, base64, os, torch
from flask import Flask, request, jsonify, Response
from PIL import Image
from models import load_generators, translate, is_sketch

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G_AB, G_BA, SIZE = load_generators(os.path.join(HERE, 'generators.pth'), device)
app = Flask(__name__)

def b64(img):
    buf = io.BytesIO(); img.save(buf, 'PNG'); return base64.b64encode(buf.getvalue()).decode()

HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CycleGAN Face <-> Sketch</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;text-align:center}
h1{margin:0 0 4px}p.sub{color:#94a3b8;margin-top:0}
.card{background:#1e293b;border-radius:14px;padding:18px;max-width:900px;margin:16px auto}
button,select{background:#6366f1;color:#fff;border:0;border-radius:8px;padding:10px 16px;margin:4px;font-size:15px;cursor:pointer}
select{background:#334155}input[type=file]{display:none}
.row{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
.box{flex:1;min-width:260px}.box img,video{width:100%;max-width:340px;border-radius:10px;background:#0b1220;min-height:120px}
#status{margin:10px;color:#a5b4fc;min-height:22px}
</style></head><body>
<h1>Face &harr; Sketch (CycleGAN)</h1>
<p class="sub">Upload a picture or use your camera. The app detects whether it is a sketch or a real face and converts it.</p>
<div class="card">
  <button onclick="document.getElementById('f').click()">Upload image</button>
  <input id="f" type="file" accept="image/*">
  <button onclick="startCam()">Use camera</button>
  <button id="snap" onclick="snap()" style="display:none">Capture &amp; convert</button>
  <select id="mode"><option value="auto">Auto-detect</option><option value="photo">Force: input is a photo</option><option value="sketch">Force: input is a sketch</option></select>
  <div><video id="v" autoplay playsinline style="display:none"></video><canvas id="c" style="display:none"></canvas></div>
  <div id="status"></div>
  <div class="row">
    <div class="box"><h3 id="t1">Input</h3><img id="in"></div>
    <div class="box"><h3 id="t2">Output</h3><img id="out"></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
async function send(blob){
  $('status').textContent='Converting...';
  $('in').src=URL.createObjectURL(blob);
  const fd=new FormData(); fd.append('image',blob,'img.png'); fd.append('mode',$('mode').value);
  try{
    const r=await fetch('/convert',{method:'POST',body:fd}); const j=await r.json();
    if(j.error){$('status').textContent=j.error;return;}
    $('out').src='data:image/png;base64,'+j.output;
    $('t1').textContent='Input (detected: '+j.detected+')'; $('t2').textContent='Output ('+j.target+')';
    $('status').textContent='Done';
  }catch(e){$('status').textContent='Request failed: '+e;}
}
$('f').onchange=e=>{if(e.target.files[0])send(e.target.files[0]);};
async function startCam(){
  try{const s=await navigator.mediaDevices.getUserMedia({video:true});
    $('v').srcObject=s;$('v').style.display='inline-block';$('snap').style.display='inline-block';}
  catch(e){$('status').textContent='Camera unavailable (needs HTTPS and permission).';}
}
function snap(){
  const v=$('v'),c=$('c');c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);c.toBlob(b=>send(b),'image/png');
}
</script></body></html>"""

@app.route('/')
def index(): return Response(HTML, mimetype='text/html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files: return jsonify(error='No image received')
    try: img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception: return jsonify(error='Could not read the image')
    mode = request.form.get('mode', 'auto')
    sketch = is_sketch(img) if mode == 'auto' else (mode == 'sketch')
    out = translate(G_BA if sketch else G_AB, img, SIZE).resize((256, 256), Image.BICUBIC)
    return jsonify(output=b64(out), detected='sketch' if sketch else 'real face',
                   target='real face' if sketch else 'sketch')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
