# Voice Clone Website (VoxCPM2) - Google Colab
# Colab loader cell ကနေ exec လုပ်ပြီး run မယ့်ဖိုင်
import subprocess, sys, os, threading, uuid, re, logging, tempfile

def sh(c):
    subprocess.run(c, shell=True, check=True)

try:
    import voxcpm, flask, soundfile
except ImportError:
    print("⏳ Install လုပ်နေသည်...")
    sh("pip install -q voxcpm flask soundfile")

import numpy as np, soundfile as sf, torch
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.serving import make_server
from google.colab import output

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ GPU မရှိ (Runtime > Change runtime type > T4)")
print("⏳ Model load လုပ်နေသည် (ပထမအကြိမ် မိနစ်အနည်းငယ်ကြာမည်)...")
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = model.tts_model.sample_rate
print("✅ Model အဆင်သင့်")

WORK = tempfile.mkdtemp()
jobs = {}
LOCK = threading.Lock()

STEPS = ["အသံဖိုင် လက်ခံခြင်း", "စာကို အပိုင်းခွဲခြင်း", "အသံ ထုတ်လုပ်ခြင်း",
         "အမြန်နှုန်း ချိန်ခြင်း", "MP3 ပြောင်းခြင်း"]

# အရည်အသွေး: (inference_timesteps, cfg_value)
QUALITY = {"normal": (6, 2.0), "good": (10, 2.0), "best": (20, 2.2)}


def split_text(text, maxlen=160):
    text = text.replace("\r", "")
    parts = re.split(r'(?<=[။!?\.\n])', text)
    out, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > maxlen * 1.6:          # အရှည်ကြီးရင် ၊ သို့ space နဲ့ ထပ်ခွဲ
            cut = max(p.rfind("၊", 0, maxlen), p.rfind(" ", 0, maxlen))
            if cut < 20:
                cut = maxlen
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if len(cur) + len(p) + 1 <= maxlen:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return [x for x in out if x]


def run_job(jid, src, text, speed, q):
    J = jobs[jid]

    def upd(step, pct, msg=""):
        J.update(step=step, pct=round(pct, 1), msg=msg)

    try:
        with LOCK:
            # 1) reference အသံ
            upd(0, 2, "အသံဖိုင် ပြင်ဆင်နေသည်")
            ref = f"{WORK}/{jid}_ref.wav"
            sh(f'ffmpeg -y -loglevel error -i "{src}" -t 60 -ac 1 -ar 16000 "{ref}"')
            # 2) စာခွဲ
            upd(1, 5, "စာခွဲနေသည်")
            chunks = split_text(text)
            if not chunks:
                raise ValueError("စာမရှိပါ")
            steps_, cfg = QUALITY[q]
            # 3) အသံထုတ်
            wavs = []
            n = len(chunks)
            for i, c in enumerate(chunks):
                upd(2, 8 + 77 * i / n, f"အပိုင်း {i+1}/{n}")
                w = model.generate(text=c, reference_wav_path=ref,
                                   cfg_value=cfg, inference_timesteps=steps_)
                wavs.append(np.asarray(w, dtype=np.float32))
                wavs.append(np.zeros(int(0.25 * SR), dtype=np.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            full = np.concatenate(wavs)
            raw = f"{WORK}/{jid}_raw.wav"
            sf.write(raw, full, SR)
            # 4) speed (pitch မပြောင်းဘဲ)
            upd(3, 88, f"အမြန်နှုန်း x{speed}")
            sped = f"{WORK}/{jid}_sp.wav"
            if abs(speed - 1.0) > 0.01:
                sh(f'ffmpeg -y -loglevel error -i "{raw}" -filter:a "atempo={speed}" "{sped}"')
            else:
                sped = raw
            # 5) mp3
            upd(4, 94, "MP3 ပြောင်းနေသည်")
            mp3 = f"{WORK}/{jid}.mp3"
            sh(f'ffmpeg -y -loglevel error -i "{sped}" -codec:a libmp3lame -qscale:a 2 "{mp3}"')
            J.update(step=5, pct=100, msg="ပြီးပါပြီ", done=True, file=mp3)
    except Exception as e:
        J.update(error=str(e))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/start", methods=["POST"])
def start():
    f = request.files.get("audio")
    text = (request.form.get("text") or "").strip()
    if not f or not text:
        return jsonify(error="အသံနဲ့ စာ နှစ်ခုလုံး ထည့်ပါ"), 400
    speed = min(2.0, max(0.5, float(request.form.get("speed", 1))))
    q = request.form.get("quality", "good")
    if q not in QUALITY:
        q = "good"
    jid = uuid.uuid4().hex[:10]
    src = f"{WORK}/{jid}_src"
    f.save(src)
    jobs[jid] = dict(step=0, pct=0, msg="စောင့်ဆိုင်းနေသည်", done=False, error=None, file=None)
    threading.Thread(target=run_job, args=(jid, src, text, speed, q), daemon=True).start()
    return jsonify(id=jid)


@app.route("/status/<jid>")
def status(jid):
    J = jobs.get(jid)
    if not J:
        return jsonify(error="မတွေ့ပါ"), 404
    return jsonify({k: J[k] for k in ("step", "pct", "msg", "done", "error")})


@app.route("/audio/<jid>")
def audio(jid):
    J = jobs.get(jid)
    if not J or not J.get("file"):
        return "no", 404
    return send_file(J["file"], mimetype="audio/mpeg", conditional=True)


HTML = r'''<!DOCTYPE html><html lang="my"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;background:#0f1220;color:#e9ecf5;font-family:system-ui,"Noto Sans Myanmar",sans-serif;padding:14px}
.card{background:#181c30;border:1px solid #2a3050;border-radius:14px;padding:14px;margin-bottom:12px}
h1{font-size:20px;margin:0 0 12px}h3{margin:0 0 8px;font-size:15px;color:#9db4ff}
textarea{width:100%;height:150px;background:#0f1220;color:#fff;border:1px solid #333b66;border-radius:10px;padding:10px;font-size:15px;resize:vertical}
input[type=file]{width:100%;color:#cfd6ff}audio{width:100%;margin-top:8px}
input[type=range]{width:100%}
.q{display:flex;gap:8px}.q button{flex:1;padding:10px 4px;border-radius:10px;border:1px solid #333b66;background:#0f1220;color:#cfd6ff;font-size:14px}
.q button.on{background:#4c6fff;border-color:#4c6fff;color:#fff}
#go{width:100%;padding:14px;border:0;border-radius:12px;background:linear-gradient(90deg,#4c6fff,#8a4cff);color:#fff;font-size:17px;font-weight:700}
#go:disabled{opacity:.5}
.bar{height:12px;background:#0f1220;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#33d17a,#4c6fff);transition:width .4s}
.st{display:flex;align-items:center;gap:10px;padding:6px 0;color:#7e87b5;font-size:15px}
.st.cur{color:#fff}.st.ok{color:#5ee08f}
.ic{width:22px;height:22px;border-radius:50%;border:2px solid #3a4272;display:flex;align-items:center;justify-content:center;font-size:13px;flex:none}
.st.ok .ic{background:#33d17a;border-color:#33d17a;color:#fff}
.st.cur .ic{border-color:#4c6fff;border-top-color:transparent;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
a.dl{display:block;text-align:center;margin-top:10px;padding:12px;border-radius:10px;background:#33d17a;color:#04210f;font-weight:700;text-decoration:none}
.err{color:#ff7b7b;margin-top:8px}.sm{font-size:13px;color:#8d96c4}
</style></head><body>
<h1>🎙️ Voice Clone</h1>

<div class="card"><h3>၁။ အသံထည့်ပါ (၁၀-၃၀ စက္ကန့် ရှင်းရှင်း)</h3>
<input type="file" id="f" accept="audio/*,video/*"><audio id="refp" controls style="display:none"></audio></div>

<div class="card"><h3>၂။ စာထည့်ပါ (အကန့်အသတ်မရှိ)</h3>
<textarea id="t" placeholder="ဒီမှာ စာရေးပါ..."></textarea><div class="sm" id="cnt">0 လုံး</div></div>

<div class="card"><h3>၃။ အသံအမြန်နှုန်း: <b id="sv">1.0</b>x</h3>
<input type="range" id="sp" min="0.5" max="1.5" step="0.05" value="1">
<div class="sm">0.5 = နှေး · 1.0 = ပုံမှန် · 1.5 = မြန်</div>
<h3 style="margin-top:12px">အသံအရည်အသွေး</h3>
<div class="q"><button data-q="normal">ပုံမှန်</button><button data-q="good" class="on">ကောင်း</button><button data-q="best">အကောင်းဆုံး</button></div>
<div class="sm" id="qh">ကောင်း = မျှတသည်</div></div>

<button id="go">▶ စတင်မည်</button>

<div class="card" id="pc" style="display:none;margin-top:12px">
<div style="display:flex;justify-content:space-between;margin-bottom:6px"><b>လုပ်ဆောင်နေသည်</b><b id="pct">0%</b></div>
<div class="bar"><i id="bar"></i></div><div class="sm" id="msg" style="margin:6px 0"></div>
<div id="steps"></div><div class="err" id="err"></div></div>

<div class="card" id="rc" style="display:none"><h3>✅ ပြီးပါပြီ — စမ်းနားထောင်ပါ</h3>
<audio id="out" controls></audio><a class="dl" id="dl" href="#">⬇ MP3 ဒေါင်းလုတ်</a></div>

<script>
const STEPS=["အသံဖိုင် လက်ခံခြင်း","စာကို အပိုင်းခွဲခြင်း","အသံ ထုတ်လုပ်ခြင်း","အမြန်နှုန်း ချိန်ခြင်း","MP3 ပြောင်းခြင်း"];
const $=id=>document.getElementById(id);let q="good",jid=null,timer=null;
const QH={normal:"ပုံမှန် = အမြန်ဆုံး",good:"ကောင်း = မျှတသည်",best:"အကောင်းဆုံး = အရည်အသွေးမြင့်၊ အချိန်ပိုကြာသည်"};
$("f").onchange=e=>{const f=e.target.files[0];if(f){$("refp").src=URL.createObjectURL(f);$("refp").style.display="block"}};
$("t").oninput=()=>$("cnt").textContent=$("t").value.length+" လုံး";
$("sp").oninput=()=>$("sv").textContent=parseFloat($("sp").value).toFixed(2).replace(/0$/,"");
document.querySelectorAll(".q button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".q button").forEach(x=>x.classList.remove("on"));b.classList.add("on");q=b.dataset.q;$("qh").textContent=QH[q]});
function render(s){
 $("pct").textContent=Math.floor(s.pct)+"%";$("bar").style.width=s.pct+"%";$("msg").textContent=s.msg||"";
 $("steps").innerHTML=STEPS.map((n,i)=>{const c=i<s.step?"ok":(i==s.step?"cur":"");
  return `<div class="st ${c}"><div class="ic">${c=="ok"?"✓":""}</div>${n}</div>`}).join("");
}
$("go").onclick=async()=>{
 const f=$("f").files[0],t=$("t").value.trim();
 $("err").textContent="";
 if(!f||!t){alert("အသံနဲ့ စာ နှစ်ခုလုံး ထည့်ပါ");return}
 const fd=new FormData();fd.append("audio",f);fd.append("text",t);fd.append("speed",$("sp").value);fd.append("quality",q);
 $("go").disabled=true;$("rc").style.display="none";$("pc").style.display="block";render({step:0,pct:0,msg:"တင်နေသည်..."});
 const r=await fetch("start",{method:"POST",body:fd});const j=await r.json();
 if(j.error){$("err").textContent=j.error;$("go").disabled=false;return}
 jid=j.id;
 timer=setInterval(async()=>{
  const s=await (await fetch("status/"+jid)).json();
  if(s.error){clearInterval(timer);$("err").textContent="❌ "+s.error;$("go").disabled=false;return}
  render(s);
  if(s.done){clearInterval(timer);$("go").disabled=false;
   const u="audio/"+jid;$("out").src=u;$("rc").style.display="block";
   $("dl").onclick=async e=>{e.preventDefault();const b=await (await fetch(u)).blob();
    const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="voice_clone.mp3";a.click()};
  }
 },700);
};
</script></body></html>'''

# Server စတင် (ပြန် run ရင် အဟောင်းပိတ်)
if globals().get("_srv"):
    try:
        _srv.shutdown()
    except Exception:
        pass
_srv = make_server("0.0.0.0", 8000, app, threaded=True)
threading.Thread(target=_srv.serve_forever, daemon=True).start()

print("✅ Website အောက်မှာ ပေါ်လာပါပြီ 👇")
output.serve_kernel_port_as_iframe(8000, height=1000)
