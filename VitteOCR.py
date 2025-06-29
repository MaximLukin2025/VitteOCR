# -*- coding: utf-8 -*-
"""Vitte OCR.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I9PupSTkjAt74BxWAH9T8p4DHY70p0WT
"""

# ==============================================================
# 0. Установка зависимостей
# ==============================================================
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr libtesseract-dev fonts-dejavu-core
!wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata \
       -O /usr/share/tesseract-ocr/4.00/tessdata/rus.traineddata
!pip install -q openai==1.16.2 pytesseract pillow opencv-python-headless \
                Levenshtein tqdm matplotlib

# ==============================================================
# 1. Импорт пакетов
# ==============================================================
import os, io, json, base64, zipfile, re, random, textwrap
from pathlib import Path
from typing import List, Dict
import cv2, numpy as np, pytesseract, openai, Levenshtein as Lev
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================
# 2. OpenAI Proxy-клиент
# ==============================================================
PROXY_BASE = "https://api.proxyapi.ru/openai/v1"
API_KEY    = "sk-2uHtBOkjr3ZrCn43aUt4WdEZ20JaXu49"   # замените на свой
client = openai.OpenAI(api_key=API_KEY, base_url=PROXY_BASE)

REFUSAL_TRIGGERS = ("I’m sorry", "I'm sorry", "I cannot", "I can't", "enable")

def safe_gpt4o(img_b64: str, raw: str = "", max_retry: int = 3) -> str:
    """JPEG-base64  →  текст.  Многократный ретрай + усечение RAW."""
    raw_short = (raw[:800] + "…") if len(raw) > 800 else raw or "<нет RAW>"
    prompts = [
        f"Черновой OCR:\n«{raw_short}».\nИсправь символ-в-символ.",
        "Напиши текст на изображении без изменений.",
        "Скажи, какой текст виден на картинке, дословно."
    ]
    for attempt in range(max_retry):
        msg = [
            {"role": "system",
             "content": "Ты OCR-эксперт. Верни текст точно, без маскирования."},
            {"role": "user", "content": [
                {"type": "text",      "text": prompts[attempt]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
        ]
        rsp = client.chat.completions.create(
            model="gpt-4o-mini", messages=msg,
            temperature=0, max_tokens=350)
        out = rsp.choices[0].message.content.strip()
        if not any(t in out for t in REFUSAL_TRIGGERS):
            return out        # ✓
    return ""                 # отказ после 3-х попыток

def gpt4o_rules(raw:str, fixed:str, limit=5)->List[str]:
    """Извлекаем до `limit` правил s/err/corr/."""
    prompt = ("Сравни тексты и сформулируй до 5 правил вида "
              "s/ошибка/исправление/ или delete(X). "
              "RAW: «"+raw+"»\nFIXED: «"+fixed+"»")
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Выводи по одному правилу в строке."},
                  {"role":"user","content":prompt}],
        temperature=0.2, max_tokens=120
    ).choices[0].message.content
    rules=[r.strip() for r in rsp.splitlines() if r.strip()][:limit]
    return rules

# ==============================================================
# 3. Табличная адаптивная предобработка  (новизна №1)
# ==============================================================
def degrid(img_gray:np.ndarray)->np.ndarray:
    """Удаляем горизонт/вертик. линии, сохраняя текст."""
    thresh = cv2.adaptiveThreshold(img_gray,255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV,15,11)
    # горизонт
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,1))
    detect_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel,1)
    # вертикаль
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,40))
    detect_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel,1)
    mask = cv2.bitwise_or(detect_h, detect_v)
    cleaned = cv2.inpaint(img_gray, mask, 5, cv2.INPAINT_NS)
    return cleaned

def table_aware_preproc(pil:Image.Image, target_h=1200)->Image.Image:
    """CLAHE + degrid + adaptive binarize, масштаб под GPT ≤768px."""
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    gray = degrid(gray)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    bin_ = cv2.adaptiveThreshold(clahe,255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,11)
    h = target_h
    w = int(pil.width * h / pil.height)
    bin_ = cv2.resize(bin_, (w,h), cv2.INTER_CUBIC)
    return Image.fromarray(bin_)

# ==============================================================
# 4. Помощники
# ==============================================================
def to_jpeg_b64(pil: Image.Image, max_side=2048, q=90) -> str:
    w, h = pil.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        pil = pil.resize((int(w*s), int(h*s)), Image.BICUBIC)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def cer(a:str,b:str)->float:
    return Lev.distance(a,b)/max(1,len(b))

def apply_rules(text:str, rules:List[str])->str:
    for r in rules:
        if r.startswith("s/") and r.count("/")>=3:
            _,bad,good,_ = r.split("/",3)
            text=text.replace(bad,good)
        elif r.startswith("delete(") and r.endswith(")"):
            text=text.replace(r[7:-1],"")
    return text

# ==============================================================
# 5. Загрузка примеров —> DATA_DIR
# ==============================================================
DATA_DIR = Path('/content/dataset')
if not DATA_DIR.exists():
    !wget -q "https://www.dropbox.com/scl/fi/mmyjbuglk8iwvybw7qmf5/dataset.zip?rlkey=v10y9pqrqobbmy0tl0zm4l0jl&st=9go362ir&dl=1" -O dataset.zip
    with zipfile.ZipFile('dataset.zip') as zf:
        zf.extractall('/content/')
    print("Dataset extracted to", DATA_DIR)

print("Файлы для эксперимента:", list(DATA_DIR.iterdir()))

import random
from pathlib import Path

# Путь к папке с изображениями
DATA_DIR = Path('/content/dataset')

# Собираем список всех файлов (изображений) в папке
all_images = list(DATA_DIR.glob('*'))

# На случай, если файлов меньше или ровно 15
if len(all_images) <= 15:
    print(f"В папке всего {len(all_images)} файлов — удалять ничего не нужно.")
else:
    # Выбираем случайные 15 для сохранения
    to_keep = set(random.sample(all_images, 15))

    # Удаляем все файлы, не вошедшие в выборку
    removed = 0
    for img in all_images:
        if img not in to_keep:
            img.unlink()
            removed += 1

    print(f"Сохранено {len(to_keep)} изображений, удалено {removed} изображений.")

# ==============================================================
# 6. Pipeline на 2-х фазах  (OCR → GPT-корректор)
# ==============================================================
all_paths = sorted(DATA_DIR.glob("*.jpg"))
random.shuffle(all_paths)

global_rules=[]
results=[]

# -------- функция разбиения -------------
def tiles(pil: Image.Image, side=1024, overlap=64):
    w, h = pil.size
    for y in range(0, h, side - overlap):
        for x in range(0, w, side - overlap):
            yield pil.crop((x, y, min(x+side, w), min(y+side, h)))

# -------- меняем DATA_DIR чтение: -----------------
def make_tile_paths(img_path: Path) -> List[Path]:
    pil = Image.open(img_path).convert("RGB")
    t_dir = img_path.parent / ("_tiles_" + img_path.stem)
    t_dir.mkdir(exist_ok=True)
    paths = []
    for i, crop in enumerate(tiles(pil)):
        fname = t_dir / f"{img_path.stem}_{i}.jpg"
        crop.save(fname, quality=95)
        paths.append(fname)
    return paths

tile_paths = []
for p in DATA_DIR.glob("*.jpg"):
    tile_paths += make_tile_paths(p)

all_paths = tile_paths              # заменяем исходный список

for p in tqdm(all_paths, desc="OCR-LLM loop"):
    pil_raw = Image.open(p).convert("RGB")
    img_proc = table_aware_preproc(pil_raw)

    # --- динамический выбор psm ---
    density = np.mean(np.array(img_proc)<200)  # доля «черного»
    psm = 3 if density<0.05 else 6
    raw_txt = pytesseract.image_to_string(
        img_proc, lang="rus",
        config=f"--oem 1 --psm {psm} --dpi 400").strip()

    if not raw_txt:
      for psm_try in (4, 11):
          raw_txt = pytesseract.image_to_string(
              img_proc, lang="rus",
              config=f"--oem 1 --psm {psm_try} --dpi 400").strip()
          if raw_txt:
              break

    # GPT-коррекция
    img64 = to_jpeg_b64(img_proc)
    fixed_txt = safe_gpt4o(img64, raw_txt)

    # правила из GPT
    rules = gpt4o_rules(raw_txt, fixed_txt)
    global_rules.extend(rules)

    results.append(
      {"file":p.name,"raw":raw_txt,"fixed":fixed_txt,
       "rules":rules,"cer_raw":cer(raw_txt,fixed_txt)}
    )

# уникализируем rules
global_rules = list(dict.fromkeys(global_rules))
print("\nСводные правила (первые 15):")
print("\n".join(global_rules[:15]))

# ==============================================================
# 7. Применяем distilled rules на те же файлы
# ==============================================================
after=[]
for r in results:
    corr = apply_rules(r["raw"], global_rules)
    after.append(cer(corr, r["fixed"]))
print(f"\nCER до правил  {np.mean([r['cer_raw'] for r in results]):.2%}")
print(f"CER после правил {np.mean(after):.2%}")

# ==============================================================
# 8. Визуальная проверка (k примеров)
# ==============================================================
import matplotlib.pyplot as plt
from textwrap import fill
from pathlib import Path

def wrap(txt, width=70):
    """Удобная обёртка, удаляем лишние пробелы"""
    return "\n".join(fill(line, width) for line in txt.splitlines())

def resolve_img(fname: str | Path) -> Path:
    """Ищем файл среди DATA_DIR и всех его подкаталогов."""
    p = Path(fname)
    if p.is_file():          # уже абсолютный путь
        return p
    direct = DATA_DIR / p    # пробуем напрямую
    if direct.is_file():
        return direct
    matches = list(DATA_DIR.rglob(p.name))   # поиск в _tiles_*
    if matches:
        return matches[0]
    raise FileNotFoundError(fname)

def show_examples(result_list, k=2, width=80):
    k = min(k, len(result_list))
    fig, axes = plt.subplots(
        nrows=k, ncols=2, figsize=(18, 4*k),
        gridspec_kw={"width_ratios":[1.3, 2]})
    if k == 1: axes = [axes]                              # унифицируем итерацию

    for ax_row, res in zip(axes, result_list[:k]):
        # ----- левая колонка: картинка -----
        ax_img, ax_txt = ax_row
        img_path = resolve_img(res["file"])
        img = Image.open(img_path)
        ax_img.imshow(img); ax_img.axis("off")
        ax_img.set_title(res["file"], fontsize=12, pad=8)

        # ----- правая колонка: три строки текста -----
        def tag(label, txt):
            """Возвращаем кортеж: (цвет, строка)."""
            clr = {"RAW": "#e63946", "RULE": "#457b9d", "GPT": "#2a9d8f"}[label]
            return clr, f"[{label}]  {txt}"

        raw_c,  raw  = tag("RAW",  res["raw"]  or "—")
        rule_c, rule = tag("RULE", apply_rules(res["raw"], global_rules) or "—")
        gpt_c,  gpt  = tag("GPT",  res["fixed"] or "—")

        def color_wrap(txt, c):               # переносим + раскрашиваем
            return wrap(txt, width), c

        chunks = [color_wrap(raw,  raw_c),
                  color_wrap(rule, rule_c),
                  color_wrap(gpt,  gpt_c)]

        ax_txt.axis("off")
        y0 = 1.0
        for para, col in chunks:
            ax_txt.text(0, y0, para, va="top", ha="left",
                        fontsize=9, family="monospace",
                        color=col, bbox=dict(boxstyle="round,pad=0.4",
                                            facecolor="#f9f9f9", edgecolor="#cccccc"))
            y0 -= para.count("\n")*0.05 + 0.12          # сдвигаем вниз

        full_txt = f"{raw}\n\n{rule}\n\n{gpt}"
        ax_txt.axis("off")
        # фон-рамка для удобства чтения
        bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#bbbbbb")
        ax_txt.text(0, 1, full_txt, va="top", ha="left", fontsize=9, family="monospace",
                    bbox=bbox_props, wrap=True)

    plt.tight_layout(h_pad=2.0)
    plt.show()

# ---- вызов ----
show_examples(results, k=2, width=75)

ART_DIR = Path("/content/results"); ART_DIR.mkdir(exist_ok=True)
# правила
(Path(ART_DIR)/"rules.txt").write_text("\n".join(global_rules), encoding="utf8")
# сырые + фикс
json.dump({r["file"]: {"raw":r["raw"],"fixed":r["fixed"]} for r in results},
          open(ART_DIR/"labels_raw_fixed.json","w",encoding="utf8"),
          ensure_ascii=False, indent=2)
# предобработанные картинки
PREP = ART_DIR/"preprocessed"; PREP.mkdir(exist_ok=True)
for p in DATA_DIR.glob("*.jpg"):
    table_aware_preproc(Image.open(p)).save(PREP/p.name, quality=95)
print("✓ Артефакты сохранены в", ART_DIR)
