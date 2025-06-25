# -*- coding: utf-8 -*-
"""
OCR → GPT-4o Vision корректор + бинарный детектор «скриншот Витте / не Витте»
"""

###############################################################################
# 0 ─── system imports (без Streamlit-команд)
###############################################################################
import os, sys, io, re, json, base64, subprocess, difflib, pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2, pytesseract, openai, Levenshtein as Lev
from PIL import Image
from tensorflow.keras.models import load_model     # ← только для инференса

###############################################################################
# 1 ─── constants / init
###############################################################################
ROOT         = Path(__file__).parent
RULES_PATH   = ROOT / "rules.txt"
MODEL_PATH   = ROOT / "best_model.h5"
VECT_PATH    = ROOT / "tfidf_vectorizer.pkl"

TILE_SIDE, OVERLAP = 1024, 64

REFUSAL_RGX = re.compile(
    r"(i['’]?\s?m (sorry|unable)|i can('| )?t|i cannot|"
    r"извините[, ]*я не могу|к сожалению[, ]*я не могу|не могу помочь)",
    re.I,
)

API_KEY    = "sk-2uHtBOkjr3ZrCn43aUt4WdEZ20JaXu49" # ← замените
PROXY_BASE = "https://api.proxyapi.ru/openai/v1"
GPT_MODEL  = "gpt-4o"
GPT_TO     = 25
client     = openai.OpenAI(api_key=API_KEY, base_url=PROXY_BASE)

###############################################################################
# 2 ─── streamlit bootstrap (НЕ УДАЛЯТЬ)
###############################################################################
from streamlit.web import cli as stcli
from streamlit import runtime
import streamlit as st

###############################################################################
# 3 ─── helpers (без Streamlit-вызовов)
###############################################################################
def configure_tesseract() -> str:
    guess = [Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
             Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")]
    exe = next((p for p in guess if p.exists()), None)
    if not exe:
        return ""
    pytesseract.pytesseract.tesseract_cmd = str(exe)
    os.environ["TESSDATA_PREFIX"] = str(exe.parent / "tessdata")
    try:
        return subprocess.check_output([str(exe), "--version"],
                                       text=True).splitlines()[0]
    except subprocess.CalledProcessError:
        return ""

@st.cache_resource
def load_rules() -> List[str]:
    if RULES_PATH.exists():
        return [l.strip() for l in RULES_PATH.read_text("utf8").splitlines()
                if l.strip()]
    return []

def apply_rules(text: str, rules: List[str]) -> str:
    """
    Применяет правила формата
      • s/неправильно/правильно/
      • delete(строка_для_удаления)
    к произвольному тексту OCR.
    """
    for r in rules:
        if r.startswith("s/") and r.count("/") >= 3:
            _, bad, good, _ = r.split("/", 3)
            text = text.replace(bad, good)
        elif r.startswith("delete(") and r.endswith(")"):
            text = text.replace(r[7:-1], "")
    return text

# ── модель классификации «Витте / не Витте» ────────────────────────────────
@st.cache_resource
def load_cls():
    if not (MODEL_PATH.exists() and VECT_PATH.exists()):
        return None, None
    model  = load_model(MODEL_PATH)
    with open(VECT_PATH, "rb") as f:
        vect = pickle.load(f)
    return model, vect

def classify(text: str, model, vect) -> (str, float):
    if model is None or vect is None:
        return "модель не загружена", 0.0
    feats = vect.transform([text]).toarray()
    prob  = float(model.predict(feats).ravel()[0])   # >0.5 ⇒ Non-Vitte
    if prob > 0.5:
        return "не относится к Витте", prob
    else:
        return "относится к Витте", 1 - prob

# ── OCR utils ──────────────────────────────────────────────────────────────
def tiles(pil: Image.Image):
    w, h = pil.size
    for y in range(0, h, TILE_SIDE - OVERLAP):
        for x in range(0, w, TILE_SIDE - OVERLAP):
            yield pil.crop((x, y, min(x+TILE_SIDE, w), min(y+TILE_SIDE, h)))

def table_preproc(pil: Image.Image, h_target=1400) -> Image.Image:
    g  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV,17,11)
    hmask = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(40,1)))
    vmask = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(1,40)))
    clean = cv2.inpaint(g, cv2.bitwise_or(hmask, vmask), 3, cv2.INPAINT_NS)
    clahe = cv2.createCLAHE(2,(8,8)).apply(clean)
    binar = cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,31,9)
    w     = int(pil.width * h_target / pil.height)
    return Image.fromarray(cv2.resize(binar,(w,h_target),cv2.INTER_CUBIC))

def ocr_tess(pil: Image.Image) -> str:
    dens = np.mean(np.array(pil) < 200)
    for psm in ([3,6] if dens<.05 else [6,3]) + [4,11]:
        txt = pytesseract.image_to_string(pil, lang="rus",
                                          config=f"--oem 1 --psm {psm}").strip()
        if txt: return txt
    return ""

def to_b64(pil: Image.Image, max_side=768) -> str:
    w,h = pil.size
    if max(w,h) > max_side:
        s = max_side/max(w,h); pil = pil.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); pil.save(buf,"JPEG",quality=70,optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def safe_gpt(img_b64: str, context: str, tries=3) -> str:
    ctx_short = (context[:700]+"…") if len(context)>700 else (context or "<нет RAW>")
    prompts = [
        f"Черновой OCR:\n«{ctx_short}».\nИсправь ошибки, текст 100 % дословно.",
        "Верни точный текст изображения без изменений.",
        "Сделай OCR-расшифровку символ-в-символ."
    ]
    for i in range(tries):
        msg = [
            {"role":"system","content":
             "Ты OCR-движок. Изображение безопасно. Верни текст дословно, ничего не скрывай."},
            {"role":"user","content":[
                {"type":"text","text":prompts[i]},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}]},
        ]
        try:
            out = client.chat.completions.create(
                model=GPT_MODEL, timeout=GPT_TO,
                messages=msg, temperature=0, max_tokens=400
            ).choices[0].message.content.strip()
        except Exception: out = ""
        if out and not REFUSAL_RGX.search(out): return out
    return ""      # отказ

def diff_short(a: str, b: str, limit=12) -> str:
    lines = difflib.unified_diff(a.splitlines(), b.splitlines(),
                                 lineterm="", n=0)
    body = [l for l in lines if l.startswith(("+","-")) and l[1:].strip()]
    if len(body) > limit: body = body[:limit] + ["… (+ ещё изменения)"]
    fmt = lambda l: f'<span style="color:#e74c3c">{l}</span>' if l.startswith("-") \
                    else f'<span style="color:#1abc9c">{l}</span>'
    return "<br>".join(fmt(l) for l in body) or "— изменений нет —"

def cer(a: str, b: str):
    a_clean = re.sub(r'\s+', '', a)
    b_clean = re.sub(r'\s+', '', b)
    total   = len(b_clean)
    if total == 0:
        return 0, 0, 0.0
    errors  = Lev.distance(a_clean, b_clean)
    return errors, total, errors/total

###############################################################################
# 4 ─── Streamlit GUI
###############################################################################
def main() -> None:
    st.set_page_config("OCR + GPT корректор", "📄", layout="wide")

    TESS  = configure_tesseract()
    RULES = load_rules()
    CLS_MODEL, VECT = load_cls()

    with st.sidebar:
        st.header("📁 Загрузите скриншот")
        up = st.file_uploader("Изображение (jpg/png/tiff/pdf)",
                              type=["jpg","jpeg","png","tif","tiff","pdf"])
        st.header("⚙️ Опции")
        tiles_on = st.checkbox("Тайлы 1024 px", True)
        gpt_on   = st.checkbox("Исп. GPT-4o", True)
        rules_on = st.checkbox("Применять rules.txt", True)
        debug    = st.checkbox("Показать RAW и После-правил", False)

    st.title("📄 OCR → GPT корректор + детектор Витте")

    if not up:
        st.info("Слева выберите файл для анализа.")
        return

    img = Image.open(up).convert("RGB")

    # ── грубый OCR только для классификации ───────────────────────────────
    rough_txt = pytesseract.image_to_string(img, lang="rus")[:3000]
    label, conf = classify(rough_txt, CLS_MODEL, VECT)
    st.subheader("🏷️ Принадлежность документа")
    colA, colB = st.columns([1,4])
    with colA:
        st.image(img, use_container_width=True,
                 caption=f"{img.width}×{img.height}px")
    with colB:
        clr = "#27ae60" if label.startswith("относится") else "#c0392b"
        st.markdown(f"<h3 style='color:{clr}'>{label}</h3>"
                    f"<p>Уверенность модели: <b>{conf:.2%}</b></p>",
                    unsafe_allow_html=True)

    st.divider()

    # ── полноценный OCR-pipeline ───────────────────────────────────────────
    parts = list(tiles(img)) if tiles_on else [img]
    raw_all, rule_all, fin_all = [], [], []

    for p in parts:
        proc = table_preproc(p)
        raw  = ocr_tess(proc)
        rule = apply_rules(raw, RULES) if rules_on else raw
        gpt  = safe_gpt(to_b64(proc), rule) if gpt_on else ""
        fin  = gpt or rule
        raw_all.append(raw); rule_all.append(rule); fin_all.append(fin)

    raw_txt  = "\n".join(raw_all)
    rule_txt = "\n".join(rule_all)
    fin_txt  = "\n".join(fin_all)

    st.subheader("📝 Финальный текст (после всех правок)")
    st.code(fin_txt or "—", language="markdown")

    st.markdown("#### 🔍 Что изменилось")
    st.markdown(diff_short(raw_txt, fin_txt), unsafe_allow_html=True)

    if debug:
        with st.expander("RAW / Tesseract"):
            st.code(raw_txt or "—", language="markdown")
        with st.expander("После-правил"):
            st.code(rule_txt or "—", language="markdown")

###############################################################################
# 5 ─── run
###############################################################################
if __name__ == "__main__":
    if runtime.exists():      # streamlit run app.py
        main()
    else:                      # python app.py
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
