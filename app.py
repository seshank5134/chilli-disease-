"""Local Streamlit chilli-leaf prediction interface."""
from __future__ import annotations

import csv
import html
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
METRICS_PATH = ROOT / "reports" / "training_metrics.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

st.set_page_config(page_title="Chilli Leaf | AI Screening", page_icon="🌶️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#d7e6da; --muted:#9aada1; --green:#4edea3; --surface:#09160f; --panel:#111e17; --line:#2b3a31; }
.stApp { background:radial-gradient(ellipse at 86% 0%, #143524 0%, transparent 34%), #09160f; color:var(--ink); }
.block-container { max-width:1450px; padding-top:1.4rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--ink)!important; letter-spacing:-.035em; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111e17,#0b1710); border-right:1px solid #26352b; }
[data-testid="stSidebar"] h3 { letter-spacing:-.02em; }
.hero { background:linear-gradient(115deg,#14271d,#172b20 62%,#1d3b2b); border:1px solid #34533d; border-radius:16px; padding:1.65rem 1.9rem; margin:.3rem 0 1.1rem; box-shadow:0 14px 40px #0003; position:relative; overflow:hidden; }
.hero:after { content:'✳'; position:absolute; right:35px; top:-42px; font-size:190px; color:#4edea30c; }
.raceway-card { margin:.15rem 0 1.25rem; padding:1rem 1.2rem; border:1px solid #32513d; border-radius:16px; background:linear-gradient(125deg,rgba(19,39,27,.88),rgba(10,24,16,.74)); box-shadow:inset 0 1px #ffffff0b,0 14px 34px #0003; backdrop-filter:blur(18px); }
.raceway-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
.raceway-title { display:flex; align-items:center; gap:.7rem; font-weight:700; color:#e4f1e7; letter-spacing:-.015em; }
.raceway-title .flag { display:grid; place-items:center; width:2rem; height:2rem; border:1px solid #4edea344; border-radius:10px; background:#4edea318; color:#4edea3; }
.raceway-sub { color:#9eafa2; font-size:.78rem; margin-top:.18rem; }
.raceway-online { display:flex; align-items:center; gap:.45rem; color:#4edea3; font:600 .68rem/1 monospace; letter-spacing:.08em; }
.raceway-online i { width:7px; height:7px; background:#4edea3; border-radius:50%; box-shadow:0 0 12px #4edea3; }
.raceway-track { height:54px; position:relative; margin-top:.8rem; border:1px solid #314638; border-radius:10px; overflow:hidden; background:linear-gradient(90deg,#06120b,#0a1b11 48%,#06120b); }
.raceway-track:before { content:''; position:absolute; left:0; right:0; top:50%; border-top:1px dashed #4edea35c; }
.raceway-track:after { content:'INTERACTIVE SCREEN-WIDE ROAM'; position:absolute; right:12px; bottom:5px; color:#82968a; font:600 8px monospace; letter-spacing:.12em; }
.raceway-speeds { display:flex; align-items:center; gap:.35rem; border:1px solid #32483a; background:#08150eaa; border-radius:9px; padding:.22rem; }
.raceway-speeds button { color:#adbaaf; border:0; background:transparent; border-radius:6px; padding:.42rem .58rem; cursor:pointer; font:600 .68rem monospace; transition:.16s ease; }
.raceway-speeds button:hover,.raceway-speeds button[aria-pressed="true"] { color:#06170f; background:#4edea3; }
@media (max-width:700px) { .raceway-sub { max-width:220px; } .raceway-speeds button { padding:.38rem .42rem; } .raceway-track { height:42px; } }
.race-bento { background:linear-gradient(145deg,rgba(27,48,35,.78),rgba(14,27,19,.84)); border:1px solid #3a5240!important; backdrop-filter:blur(16px); box-shadow:inset 0 1px #ffffff0a,0 10px 25px #0002!important; }

.hero h1 { color:#e5f1e8!important; font-size:2.05rem; margin:0 0 .35rem; }
.hero p { color:#aebfb3; font-size:1rem; margin:0; max-width:760px; }
.eyebrow { color:#4edea3; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:700; margin-bottom:.6rem; }
.section-kicker { color:#4edea3; font-size:.72rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; margin:.3rem 0 .25rem; }
.panel { background:linear-gradient(145deg,rgba(27,48,35,.82),rgba(14,27,19,.86)); border:1px solid #385441; border-radius:14px; padding:1.05rem 1.15rem; box-shadow:inset 0 1px #ffffff0a,0 10px 28px #0003; backdrop-filter:blur(16px); }
div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,49,36,.82),rgba(13,27,19,.82)); border:1px solid #385441; padding:.95rem 1rem; border-radius:14px; box-shadow:inset 0 1px #ffffff0a,0 10px 28px #0003; backdrop-filter:blur(16px); }
div[data-testid="stMetricLabel"] { color:#a4b4a9; }
div[data-testid="stMetricValue"] { color:#e4f1e7; }
.diagnosis-banner { padding:1rem 1.05rem; margin:.2rem 0 .75rem; border:1px solid #4edea355; border-radius:13px; background:linear-gradient(130deg,#173a29,#10231a); box-shadow:inset 0 1px #ffffff0d,0 8px 22px #0002; }
.diagnosis-kicker { color:#75e8b3; font:700 .67rem monospace; letter-spacing:.12em; text-transform:uppercase; }
.diagnosis-name { color:#e5f4e8; font-size:1.45rem; font-weight:750; letter-spacing:-.025em; line-height:1.25; margin:.32rem 0 .5rem; }
.focus-note { color:#9fb3a5; font-size:.76rem; line-height:1.45; padding:.6rem .7rem; border:1px solid #35483a; border-radius:9px; background:#0a1710aa; }
.probability-list { display:flex; flex-direction:column; gap:.72rem; margin-top:.5rem; }
.probability-label { display:flex; align-items:center; justify-content:space-between; gap:.6rem; color:#cfddd2; font-size:.79rem; }
.probability-label strong { color:#e8f5eb; font:650 .78rem monospace; }
.probability-track { height:7px; overflow:hidden; border-radius:99px; background:#293a30; margin-top:.34rem; }
.probability-track span { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#10b981,#6ffbbe); box-shadow:0 0 10px #4edea344; }
.image-pair-caption { color:#94a99a; font:600 .67rem monospace; text-transform:uppercase; letter-spacing:.09em; margin:.1rem 0 .35rem; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #2b3a31; }
.stTabs [data-baseweb="tab"] { background:#111e17; border-radius:9px 9px 0 0; padding:.65rem 1rem; color:#a8b8ac; }
.stTabs [aria-selected="true"] { background:#1c3427!important; color:#4edea3!important; }
.stButton button[kind="primary"],.stDownloadButton button { background:#10a870; border-color:#10a870; color:#062116; border-radius:9px; font-weight:700; }
.stButton button[kind="primary"]:hover,.stDownloadButton button:hover { background:#4edea3; border-color:#4edea3; color:#062116; }
div[data-testid="stFileUploader"] { background:linear-gradient(145deg,rgba(24,43,31,.85),rgba(11,24,17,.9)); border:1px dashed #55735b; border-radius:12px; padding:.55rem; backdrop-filter:blur(14px); }
div[data-testid="stExpander"] { border-color:#304337; border-radius:10px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(27,48,35,.72),rgba(12,25,17,.8)); border:1px solid #385441; border-radius:15px; box-shadow:inset 0 1px #ffffff09,0 10px 28px #0002; backdrop-filter:blur(14px); }
[data-testid="stDataFrame"] { border:1px solid #2b3a31; border-radius:10px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Unsupported model architecture in checkpoint.")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prep = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization_mean"], checkpoint["normalization_std"]),
    ])
    return model, checkpoint["class_names"], prep, device


def predict_bytes(model, classes, prep, device, payload: bytes, with_focus: bool = False):
    with Image.open(io.BytesIO(payload)) as raw:
        image = raw.convert("RGB")
    tensor = prep(image).unsqueeze(0).to(device)
    focus_image = None
    if with_focus:
        activations = {}
        target_layer = model.features[-1]
        hook = target_layer.register_forward_hook(lambda _module, _inputs, output: activations.__setitem__("value", output))
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(tensor)
                best_index = int(logits[0].argmax().detach())
                gradients = torch.autograd.grad(logits[0, best_index], activations["value"])[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                heat = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
                heat = torch.nn.functional.interpolate(heat, size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
                heat = heat.detach()
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                probs = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
                heat = heat.cpu().numpy()
        finally:
            hook.remove()
        red = np.clip(1.65 * heat - .28, 0, 1)
        green = np.clip(1.15 - np.abs(heat - .5) * 2.2, 0, 1)
        blue = np.clip(1.05 - 1.35 * heat, 0, 1)
        colors = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
        alpha = (np.power(heat, .75) * 165).clip(0, 165).astype(np.uint8)
        overlay = Image.fromarray(np.dstack([colors, alpha]), mode="RGBA")
        focus_image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    else:
        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    return image, classes[order[0]], probs[order[0]], probs, order, focus_image


def field_suggestion(label, growth_stage, spread, affected_percent, noticed, observations, notes):
    guidance = {
        "Bacterial Spot": (
            "This class usually refers to dark leaf spots that may have lighter centers and darker margins. A photo classifier cannot confirm the pathogen.",
            "Check several leaves and nearby plants. TNAU recommends field sanitation and disease-free seed; clear fallen affected debris and clean tools used on affected plants. Ask a local plant-health adviser to confirm before treatment.",
            "TNAU chilli bacterial leaf spot guidance",
            "https://agritech.tnau.ac.in/crop_protection/chilli_diseases_4.html",
        ),
        "Cercospora Leaf Spot": (
            "This class usually refers to circular brown leaf spots that can have pale gray centers and darker margins. Similar-looking spots can have other causes.",
            "Compare lesions across several leaves and note whether they are spreading. Ask a local plant-health adviser to confirm the cause before treatment.",
            "TNAU chilli Cercospora leaf spot guidance",
            "https://agritech.tnau.ac.in/crop_protection/chilli_diseases_5.html",
        ),
        "Curl Virus": (
            "This class refers to leaf curling or distortion associated with the dataset's Curl Virus label; the model does not test for a virus or identify an insect vector.",
            "Check nearby plants for similar curling and look for whiteflies, which TNAU identifies as a common vector. Ask a local adviser to confirm before removing plants or applying controls.",
            "TNAU chilli viral disease guidance",
            "https://agritech.tnau.ac.in/crop_protection/chilli_diseases_7.html",
        ),
        "Nutrition Deficiency": (
            "This class means the photo resembles examples labeled Nutrition Deficiency. It does not identify a specific missing nutrient; disease and nutrient symptoms can overlap.",
            "Get a soil test before changing fertilizer, and ask a local KVK to help interpret the result for this crop and growth stage.",
            "TNAU KVK soil and water testing guidance",
            "https://www.agritech.tnau.ac.in/kvk/kvk_faq_soil_water.html",
        ),
        "White spot": (
            "The dataset label ‘White spot’ is broad and does not identify one specific cause. The model cannot distinguish fungal, bacterial, insect, or environmental causes from this label alone.",
            "Photograph both leaf surfaces and nearby plants, then ask a local plant-health adviser to confirm the cause before treatment.",
            None,
            None,
        ),
        "Healthy Leaf": (
            "The model did not find a strong match to one of its trained disease classes in this photo. This does not rule out early or unrepresented problems.",
            "Continue routine monitoring. If symptoms are present or worsen, take clear photos of several leaves and request local confirmation.",
            None,
            None,
        ),
    }
    description, suggestion, source_title, source_url = guidance.get(label, (
        "The model returned a class without an available description.",
        "Collect a few clear photos and request local expert confirmation.",
        None,
        None,
    ))
    context = f"Context recorded: {growth_stage}; {spread.lower()}; about {affected_percent}% of leaves affected; first noticed {noticed.lower()}."
    if spread in {"Several plants", "Many plants"} or noticed == "More than a week ago":
        next_step = "Because you report that symptoms affect multiple plants or have persisted, arrange a field assessment and check nearby plants."
    else:
        next_step = "If symptoms remain localized, record another clear photo in a few days and note whether this plant or nearby plants change."
    if observations:
        context += " Observed signs: " + ", ".join(observations) + "."
    if notes.strip():
        context += " Farmer note: " + notes.strip()[:250]
    return description, suggestion, next_step, context, source_title, source_url


def render_probability_bars(classes, probabilities, order):
    rows = []
    for idx in order:
        label = html.escape(classes[idx])
        percent = max(0.0, min(100.0, probabilities[idx] * 100.0))
        rows.append(
            f'<div class="probability-row"><div class="probability-label"><span>{label}</span><strong>{percent:.1f}%</strong></div>'
            f'<div class="probability-track"><span style="width:{percent:.2f}%"></span></div></div>'
        )
    st.markdown('<div class="probability-list">' + ''.join(rows) + '</div>', unsafe_allow_html=True)


def safe_zip_images(zip_payload: bytes):
    result = {}
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.is_absolute() or ".." in path.parts or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            result[path.as_posix()] = archive.read(info)
    return result


def resolve_csv_image(requested: str, images: dict[str, bytes]):
    path = PurePosixPath(requested.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a relative image path in the CSV.")
    name = path.as_posix()
    if name.startswith("./"):
        name = name[2:]
    if name in images:
        return images[name]
    matches = [data for key, data in images.items() if PurePosixPath(key).name.casefold() == PurePosixPath(name).name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("CSV filename matches more than one ZIP image; include its relative folder path.")
    raise FileNotFoundError("No matching image was found in the ZIP file.")


def csv_download(rows, default_fields):
    buf = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else default_fields
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


st.markdown('<div class="hero"><div class="eyebrow">ChilliGuard AI · Local research console</div><h1>Disease Screening</h1><p>Run the trained chilli leaf classifier on one photo or a batch. Review class scores and export prediction results for analysis.</p></div>', unsafe_allow_html=True)

RACEWAY_PANEL_HTML = r'''
<style>
#chilli-race-overlay { position:fixed; inset:0; z-index:9999; pointer-events:none; overflow:hidden; }
#chilli-race-car { position:absolute; left:0; top:0; width:142px; height:62px; pointer-events:auto; cursor:pointer; user-select:none; transform-origin:50% 50%; filter:drop-shadow(0 9px 12px #0009); transition:filter .15s; }
#chilli-race-car:hover { filter:drop-shadow(0 0 14px #4edea3a6) brightness(1.12); }
#chilli-race-car svg { width:100%; height:100%; overflow:visible; }
#chilli-race-car .wheel-rim { transform-box:fill-box; transform-origin:center; }
#chilli-race-car .speed-lines { opacity:0; transition:opacity .15s; }
#chilli-race-car.sprinting .speed-lines { opacity:.85; }
#chilli-race-hud { position:fixed; right:18px; bottom:18px; pointer-events:auto; display:flex; align-items:center; gap:9px; padding:9px 11px 9px 13px; border:1px solid #4edea355; border-radius:999px; color:#d7e6da; background:rgba(9,22,15,.88); box-shadow:0 10px 34px #0008; backdrop-filter:blur(16px); font:12px/1.2 system-ui,sans-serif; }
#chilli-race-hud .hud-icon { font-size:17px; }
#chilli-race-hud .hud-label { color:#4edea3; font-weight:700; }
#chilli-race-hud .hud-hint { color:#9aada1; font-size:10px; margin-top:2px; }
#chilli-race-hud button { padding:6px 10px; color:#4edea3; background:#4edea318; border:1px solid #4edea355; border-radius:999px; cursor:pointer; font:600 10px system-ui,sans-serif; }
#chilli-race-hud button:hover { color:#07170f; background:#4edea3; }
#chilli-race-pop { position:fixed; inset:0; pointer-events:none; }
.chilli-spark { position:absolute; width:7px; height:7px; border-radius:50%; box-shadow:0 0 12px currentColor; animation:chilli-spark .72s ease-out forwards; }
@keyframes chilli-spark { to { transform:translate(var(--dx),var(--dy)) scale(0); opacity:0; } }
@media (prefers-reduced-motion:reduce) { #chilli-race-car { transition:none; } .chilli-spark { animation-duration:.1s; } }
@media (max-width:600px) { #chilli-race-car { width:108px; height:48px; } #chilli-race-hud { right:9px; bottom:9px; transform:scale(.9); transform-origin:bottom right; } }
</style>
<section class="raceway-card" aria-label="Interactive Chilli Raceway">
  <div class="raceway-head">
    <div>
      <div class="raceway-title"><span class="flag">🏁</span><span>Chilli Raceway</span><span style="font:600 9px monospace;color:#4edea3;border:1px solid #4edea355;border-radius:5px;padding:3px 6px">UI COMPANION</span></div>
      <div class="raceway-sub">Cursor-chasing chilli racer · click the racer to pop it · use Respawn to bring it back</div>
    </div>
    <div class="raceway-speeds" role="group" aria-label="Racer speed">
      <button type="button" data-chilli-speed="0.72" aria-pressed="false">ECO</button>
      <button type="button" data-chilli-speed="1" aria-pressed="true">TURBO</button>
      <button type="button" data-chilli-speed="1.65" aria-pressed="false">MACH</button>
    </div>
  </div>
  <div class="raceway-track" aria-hidden="true"></div>
</section>
'''

RACER_OVERLAY_JS = r'''
export default function(component) {
(() => {
  if (window.__chilliRaceGame && window.__chilliRaceGame.destroy) window.__chilliRaceGame.destroy();
  const overlay = document.createElement('div');
  overlay.id = 'chilli-race-overlay';
  overlay.setAttribute('aria-label', 'Interactive chilli racer');
  document.body.appendChild(overlay);
  overlay.innerHTML = `
    <div id="chilli-race-car" role="button" tabindex="0" aria-label="Chilli racer. Click to pop." title="Hover to chase · click to pop">
      <svg viewBox="0 0 280 92" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <linearGradient id="cr-red" x1="0" x2="1" y1="0" y2=".4"><stop stop-color="#991b1b"/><stop offset=".38" stop-color="#fb4b45"/><stop offset=".76" stop-color="#dc2626"/><stop offset="1" stop-color="#7f1d1d"/></linearGradient>
          <linearGradient id="cr-leaf" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#a3f6bf"/><stop offset=".5" stop-color="#10b981"/><stop offset="1" stop-color="#047857"/></linearGradient>
          <linearGradient id="cr-glass" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#a7f3d0"/><stop offset="1" stop-color="#064e3b"/></linearGradient>
        </defs>
        <g class="speed-lines" stroke-linecap="round"><path d="M42 32H7M36 45H0M45 58H12" stroke="#4edea3" stroke-width="3"/><path d="M31 51H7" stroke="#ffb95f" stroke-width="2"/></g>
        <path d="M22 43C9 39 2 29 1 22c10 2 18 8 26 17Z" fill="url(#cr-leaf)" stroke="#4edea3"/>
        <path d="M25 47Q61 22 136 23q83-1 119 29-42 15-116 14Q60 66 25 47Z" fill="url(#cr-red)" stroke="#ffb4ab" stroke-width="1.5"/>
        <path d="M98 28q34-16 71 1l-7 12-58-1Z" fill="url(#cr-glass)" stroke="#a7f3d0"/>
        <circle cx="132" cy="30" r="5" fill="#07170f"/><circle cx="132" cy="28" r="2.4" fill="#4edea3"/>
        <path d="M228 55q23 1 31 7l-32 1Z" fill="url(#cr-leaf)"/>
        <rect x="93" y="47" width="54" height="10" rx="3" fill="#09160f" opacity=".85"/><text x="98" y="54" fill="#4edea3" font-family="monospace" font-size="6" font-weight="700">CHILLI-RUNNER</text>
        <g transform="translate(57 60)"><circle r="17" fill="#05110a" stroke="#4edea3" stroke-width="2.3"/><circle r="12" fill="#15221b" stroke="#516358"/><g class="wheel-rim" id="cr-wheel-back"><path d="M0-10V10M-10 0H10" stroke="#6ffbbe" stroke-width="2"/><circle r="3" fill="#4edea3"/></g></g>
        <g transform="translate(197 60)"><circle r="16" fill="#05110a" stroke="#4edea3" stroke-width="2.2"/><circle r="11" fill="#15221b" stroke="#516358"/><g class="wheel-rim" id="cr-wheel-front"><path d="M0-9V9M-9 0H9" stroke="#6ffbbe" stroke-width="2"/><circle r="3" fill="#4edea3"/></g></g>
      </svg>
    </div>
    <div id="chilli-race-pop"></div>
    <div id="chilli-race-hud"><span class="hud-icon">🌶️</span><div><div class="hud-label">Chilli companion</div><div class="hud-hint" id="chilli-race-mode">Cruising · hover to chase</div></div><button type="button" id="chilli-race-respawn">Respawn</button></div>`;

  const car = document.getElementById('chilli-race-car');
  const hudText = document.getElementById('chilli-race-mode');
  const wheelA = document.getElementById('cr-wheel-back');
  const wheelB = document.getElementById('cr-wheel-front');
  const particles = document.getElementById('chilli-race-pop');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const controller = new AbortController();
  let x = Math.max(16, innerWidth * .3), y = Math.max(90, innerHeight * .32);
  let vx = 1.65, vy = .95, angle = 0, spin = 0, multiplier = 1, sprintUntil = 0;
  let alive = true, raf = 0, last = performance.now(), respawnTimer = 0;
  const speedButtons = [...document.querySelectorAll('[data-chilli-speed]')];
  const mode = (value) => { if (hudText) hudText.textContent = value; };
  const setSpeed = (value, clicked) => {
    multiplier = Number(value) || 1;
    speedButtons.forEach(btn => btn.setAttribute('aria-pressed', String(btn === clicked)));
  };
  speedButtons.forEach(btn => btn.addEventListener('click', () => setSpeed(btn.dataset.chilliSpeed, btn), {signal:controller.signal}));
  const respawn = () => {
    clearTimeout(respawnTimer); alive = true; car.style.display = 'block';
    const fromLeft = Math.random() > .5;
    x = fromLeft ? -120 : innerWidth + 10;
    y = Math.max(78, Math.random() * Math.max(120, innerHeight - 190));
    vx = fromLeft ? 2.3 : -2.3; vy = (Math.random() - .5) * 2.3;
    mode('Cruising · hover to chase');
  };
  const pop = () => {
    if (!alive) return;
    alive = false; car.style.display = 'none';
    const cx = x + 70, cy = y + 30;
    const colors = ['#4edea3','#10b981','#fb4b45','#ffb95f','#ffffff'];
    for (let i=0;i<18;i++) {
      const bit = document.createElement('i');
      const a = Math.PI * 2 * i / 18 + (Math.random()-.5)*.35;
      const dist = 35 + Math.random()*55;
      bit.className = 'chilli-spark'; bit.style.left = `${cx}px`; bit.style.top = `${cy}px`;
      bit.style.color = colors[i % colors.length]; bit.style.background = colors[i % colors.length];
      bit.style.setProperty('--dx', `${Math.cos(a)*dist}px`); bit.style.setProperty('--dy', `${Math.sin(a)*dist}px`);
      particles.appendChild(bit); setTimeout(() => bit.remove(), 760);
    }
    mode('Popped · respawning'); respawnTimer = setTimeout(respawn, 1900);
  };
  car.addEventListener('click', pop, {signal:controller.signal});
  car.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pop(); } }, {signal:controller.signal});
  document.getElementById('chilli-race-respawn').addEventListener('click', respawn, {signal:controller.signal});
  window.addEventListener('mousemove', e => {
    if (!alive) return;
    const dx = x + 70 - e.clientX, dy = y + 29 - e.clientY;
    const dist = Math.hypot(dx,dy);
    if (dist < 155) {
      const d = Math.max(dist,1), force = (155-dist)/155*5 + 2.7;
      vx = dx/d*force; vy = dy/d*force; sprintUntil = performance.now()+700;
      car.classList.add('sprinting'); mode('Turbo evade!');
    }
  }, {signal:controller.signal});
  window.addEventListener('resize', () => { x = Math.min(x, innerWidth-150); y = Math.min(y, innerHeight-80); }, {signal:controller.signal});
  function tick(now) {
    if (!alive) { raf = requestAnimationFrame(tick); return; }
    const dt = Math.min((now-last)/16.67,2); last=now;
    if (!reduced) {
      if (now < sprintUntil) { vx*=.991; vy*=.991; }
      else {
        car.classList.remove('sprinting'); mode('Cruising · hover to chase');
        vx += (Math.random()-.5)*.12; vy += (Math.random()-.5)*.12;
        const speed=Math.hypot(vx,vy), max=2.2*multiplier, min=.8*multiplier;
        if(speed<min){vx=vx/speed*min;vy=vy/speed*min;} else if(speed>max){vx=vx/speed*max;vy=vy/speed*max;}
      }
      const maxX=Math.max(10,innerWidth-155), maxY=Math.max(82,innerHeight-86);
      if(x<10){x=10;vx=Math.abs(vx)+.4;} else if(x>maxX){x=maxX;vx=-Math.abs(vx)-.4;}
      if(y<78){y=78;vy=Math.abs(vy)+.4;} else if(y>maxY){y=maxY;vy=-Math.abs(vy)-.4;}
      x+=vx*dt; y+=vy*dt;
      const target=Math.atan2(vy,vx)*180/Math.PI;
      let delta=target-angle; while(delta>180)delta-=360; while(delta< -180)delta+=360;
      angle+=delta*.09*dt; spin=(spin+Math.hypot(vx,vy)*dt*12)%360;
      car.style.transform=`translate3d(${x.toFixed(1)}px,${y.toFixed(1)}px,0) rotate(${angle.toFixed(1)}deg)`;
      wheelA.style.transform=`rotate(${spin}deg)`; wheelB.style.transform=`rotate(${spin}deg)`;
    }
    raf=requestAnimationFrame(tick);
  }
  window.__chilliRaceGame = { destroy(){ controller.abort(); cancelAnimationFrame(raf); clearTimeout(respawnTimer); overlay.remove(); delete window.__chilliRaceGame; } };
  if (reduced) { car.style.transform=`translate3d(${x}px,${y}px,0)`; }
  else raf=requestAnimationFrame(tick);
})();
  return () => {
    if (window.__chilliRaceGame && window.__chilliRaceGame.destroy) window.__chilliRaceGame.destroy();
  };
}
'''

RACER_OVERLAY = st.components.v2.component(
    "chilli_race_overlay", js=RACER_OVERLAY_JS, isolate_styles=False
)

st.html(RACEWAY_PANEL_HTML)
RACER_OVERLAY()


try:
    model, classes, preprocess, device = load_classifier(str(MODEL_PATH))
except FileNotFoundError:
    st.warning("No trained model found yet. Run the training command in README.md, then refresh this page.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the trained model: {type(exc).__name__}: {exc}")
    st.stop()

metrics = {}
try:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    pass

with st.sidebar:
    st.markdown("### 🌿 ChilliGuard AI")
    st.caption("LOCAL AGRI-ML CONSOLE")
    st.divider()
    st.markdown("#### Disease screening")
    st.success("Screening model ready")
    st.caption("Upload a chilli leaf photo to view its class estimate and follow-up guidance.")
    with st.expander("Supported classes"):
        for class_name in classes:
            st.markdown(f"- {class_name}")
    st.divider()
    st.caption("Research screening only. Confirm concerns with a local agricultural expert.")

st.markdown('<div class="section-kicker">Choose a workflow</div>', unsafe_allow_html=True)
st.markdown("## Analytical Console")
tab_images, tab_batch, tab_model = st.tabs(["▣  Disease Screening", "▤  Batch Process (CSV / ZIP)", "⌁  Model Telemetry"])

with tab_images:
    st.markdown('<div class="panel">Upload one or more clear leaf photos. Results appear below and can be downloaded as a CSV.</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Leaf image files", type=[ext.lstrip(".") for ext in sorted(IMAGE_SUFFIXES)], accept_multiple_files=True, key="images")
    if uploads:
        rows = []
        for upload_index, upload in enumerate(uploads):
            with st.container(border=True):
                st.markdown(f"#### Specimen · {html.escape(upload.name)}")
                try:
                    preview, label, confidence, probabilities, order, focus = predict_bytes(model, classes, preprocess, device, upload.getvalue(), with_focus=True)
                    image_col, result_col = st.columns([1.08, 1])
                    with image_col:
                        original_col, focus_col = st.columns(2)
                        with original_col:
                            st.markdown('<div class="image-pair-caption">Original photo</div>', unsafe_allow_html=True)
                            st.image(preview, use_container_width=True)
                        with focus_col:
                            st.markdown('<div class="image-pair-caption">Model focus · Grad-CAM</div>', unsafe_allow_html=True)
                            st.image(focus, use_container_width=True)
                        st.markdown('<div class="focus-note">Highlighted areas show what influenced the image-level prediction. They are an approximate model focus map, not a verified disease outline or bounding box.</div>', unsafe_allow_html=True)
                    with result_col:
                        safe_label = html.escape(label)
                        st.markdown(f'<div class="diagnosis-banner"><div class="diagnosis-kicker">Top model prediction</div><div class="diagnosis-name">{safe_label}</div></div>', unsafe_allow_html=True)
                        st.metric("Model confidence", f"{confidence:.1%}")
                        st.markdown("**Class score distribution**")
                        render_probability_bars(classes, probabilities, order)
                    with st.expander("Farm context & follow-up suggestion", expanded=True):
                        st.caption("Optional field details tailor the follow-up suggestion only; they do not change the image model's prediction. No name or address is needed.")
                        with st.form(key=f"farm_context_{upload_index}"):
                            context_left, context_right = st.columns(2)
                            with context_left:
                                growth_stage = st.selectbox("Crop growth stage", ["Not sure", "Seedling", "Vegetative", "Flowering", "Fruiting"], key=f"growth_{upload_index}")
                                spread = st.selectbox("How widely is it seen?", ["One plant", "Several plants", "Many plants", "Not sure"], key=f"spread_{upload_index}")
                                affected_percent = st.slider("Approximate leaves affected (%)", 0, 100, 10, 5, key=f"affected_{upload_index}")
                            with context_right:
                                noticed = st.selectbox("When first noticed?", ["Today", "2–7 days ago", "More than a week ago", "Not sure"], key=f"noticed_{upload_index}")
                                observations = st.multiselect("Signs you observed", ["Leaf spots", "Leaf curling", "Yellowing", "Visible small insects", "Wilting", "Other / unsure"], key=f"observations_{upload_index}")
                                notes = st.text_input("Other field notes (optional)", max_chars=250, key=f"notes_{upload_index}")
                            submitted = st.form_submit_button("Create follow-up suggestion", type="primary")
                        if submitted:
                            description, suggestion, next_step, context_summary, source_title, source_url = field_suggestion(
                                label, growth_stage, spread, affected_percent, noticed, observations, notes
                            )
                            st.markdown("##### What this model class means")
                            st.write(description)
                            st.markdown("##### General follow-up suggestion")
                            st.info(suggestion)
                            st.write(next_step)
                            st.caption(context_summary)
                            if source_title and source_url:
                                st.markdown(f"Source: [{source_title}]({source_url})")
                            st.caption("Screening support only. Confirm the cause with a local agricultural expert before applying pesticides or fertilizer.")
                    rows.append({"filename": upload.name, "predicted_class": label, "confidence": f"{confidence:.6f}", "status": "ok"})
                except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not read or analyze this image: {exc}")
                    rows.append({"filename": upload.name, "predicted_class": "", "confidence": "", "status": f"error: {exc}"})
        st.download_button("⬇ Download image results CSV", csv_download(rows, ["filename", "predicted_class", "confidence", "status"]), "chilli_predictions.csv", "text/csv", type="primary")

with tab_batch:
    st.markdown('<div class="panel">Your CSV needs an <code>image</code>, <code>filename</code>, or <code>path</code> column. Add the matching images to a ZIP file. Relative folders are supported when filenames repeat.</div>', unsafe_allow_html=True)
    csv_col, zip_col = st.columns(2)
    with csv_col:
        csv_file = st.file_uploader("1 · Prediction list (CSV)", type=["csv"], key="csv_list")
    with zip_col:
        zip_file = st.file_uploader("2 · Images (ZIP)", type=["zip"], key="csv_zip")
    if csv_file and zip_file and st.button("Run batch predictions", type="primary", use_container_width=True):
        try:
            images = safe_zip_images(zip_file.getvalue())
            if not images:
                raise ValueError("The ZIP does not contain supported image files.")
            decoded = csv_file.getvalue().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(decoded))
            columns = reader.fieldnames or []
            image_column = next((col for col in columns if col.casefold().strip() in {"image", "filename", "path"}), None)
            if image_column is None:
                raise ValueError("CSV must include a column named image, filename, or path.")
            input_rows = list(reader)
            if not input_rows:
                raise ValueError("The CSV has a header but no data rows.")
            result_rows = []
            progress = st.progress(0.0, text="Starting predictions…")
            for index, row in enumerate(input_rows):
                image_name = (row.get(image_column) or "").strip()
                result = dict(row)
                result.update({"predicted_class": "", "confidence": "", "status": ""})
                try:
                    payload = resolve_csv_image(image_name, images)
                    _, predicted, confidence, probabilities, _, _ = predict_bytes(model, classes, preprocess, device, payload)
                    result.update({"predicted_class": predicted, "confidence": f"{confidence:.6f}", "status": "ok"})
                    for i, class_name in enumerate(classes):
                        result[f"probability_{class_name}"] = f"{probabilities[i]:.6f}"
                except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
                    result["status"] = f"error: {exc}"
                result_rows.append(result)
                progress.progress((index + 1) / len(input_rows), text=f"Processed {index + 1} of {len(input_rows)} rows")
            success_count = sum(row["status"] == "ok" for row in result_rows)
            st.success(f"Finished {len(result_rows)} rows · {success_count} images predicted · {len(result_rows) - success_count} rows need attention.")
            st.dataframe(result_rows, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download batch results CSV", csv_download(result_rows, [image_column, "predicted_class", "confidence", "status"]), "chilli_batch_predictions.csv", "text/csv", type="primary")
        except (zipfile.BadZipFile, UnicodeDecodeError, csv.Error, ValueError) as exc:
            st.error(f"Could not process these files: {exc}")

with tab_model:
    st.markdown("### Research metrics & model telemetry")
    st.caption("Technical details for project review; they are kept out of the farmer screening flow.")
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Original-only test accuracy", f"{metrics['test_original_accuracy']:.2%}" if metrics.get("test_original_accuracy") is not None else "—", help="Measured on held-out original dataset images, not augmented images.")
    metric_b.metric("Held-out test photos", metrics.get("test_original_count", "—"))
    metric_c.metric("Training images", metrics.get("train_image_count", "—"))
    metric_d.metric("Classes", len(classes))
    st.caption("Evaluation is from held-out original images. Results on photos from new devices, lighting, farms, or varieties may differ.")
    st.markdown("### Training run")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write(f"**Device:** {metrics.get('gpu_name') or metrics.get('device', 'Unknown')}")
        st.write(f"**Architecture:** {metrics.get('architecture', 'MobileNetV3-Small')}")
        st.write(f"**Best validation accuracy:** {metrics['best_validation_accuracy']:.2%}" if metrics.get("best_validation_accuracy") is not None else "**Best validation accuracy:** —")
        st.write(f"**Original-only test accuracy:** {metrics['test_original_accuracy']:.2%}" if metrics.get("test_original_accuracy") is not None else "**Original-only test accuracy:** —")
        st.write(f"**Data split:** {metrics.get('split_strategy', 'Not recorded')}")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Training history")
        history = metrics.get("history", [])
        if history:
            st.line_chart({"Training loss": {row["epoch"]: row["train_loss"] for row in history}, "Validation accuracy": {row["epoch"]: row["validation_accuracy"] for row in history}}, x_label="Epoch", y_label="Score")
        else:
            st.info("Training history is not available.")
    with st.expander("Test confusion matrix"):
        matrix = metrics.get("test_confusion_matrix")
        labels = metrics.get("confusion_matrix_labels_order", classes)
        if matrix:
            st.dataframe({"Actual class": labels, **{label: [matrix[r][c] for r in range(len(labels))] for c, label in enumerate(labels)}}, use_container_width=True, hide_index=True)
        else:
            st.info("Confusion matrix is not available.")
    st.caption("The test set contains 275 held-out original images. Dataset metrics are a useful baseline, not a guarantee of field performance.")
