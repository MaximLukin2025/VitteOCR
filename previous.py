# ==============================================================
# 0.  Установка и инициализация
# ==============================================================
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr libtesseract-dev

!pip install -q openai pillow numpy opencv-python-headless python-Levenshtein \
                tqdm matplotlib

import os, random, json, base64, re, zipfile, io, textwrap, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytesseract
import Levenshtein as Lev
import openai

# ----------  настройка Proxy API ----------
PROXY_BASE = "https://api.proxyapi.ru/openai/v1"
OPENAI_API_KEY = os.getenv("PROXY_API_KEY")  #  <<< экспортируйте перед запуском!
assert OPENAI_API_KEY, "export PROXY_API_KEY=<ваш_ключ>"

client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=PROXY_BASE)

def gpt4o(messages, model="gpt-4o-mini", max_tokens=400, T=0.0):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=T,
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

# ---------- скачиваем датасет ----------
DATA_URL = "https://www.dropbox.com/scl/fi/mmyjbuglk8iwvybw7qmf5/dataset.zip?dl=1"
ZIP = "dataset.zip"; DATA = Path("dataset")
if not DATA.exists():
    import urllib.request, tempfile, shutil
    urllib.request.urlretrieve(DATA_URL, ZIP)
    with zipfile.ZipFile(ZIP) as zf: zf.extractall()
    # корректируем вложенность, если есть
    inner = list(DATA.glob("dataset"))
    if inner:
        for f in inner[0].iterdir(): shutil.move(str(f), DATA)
        inner[0].rmdir()
print("Файлов в датасете:", len(list(DATA.glob("Screenshot_*"))))

# ==============================================================
# 1.  Вспомогательные функции
# ==============================================================
def to_jpeg_bytes(pil_img:Image.Image, max_side=768, quality=30)->bytes:
    w,h = pil_img.size
    if max(w,h) > max_side:
        scale = max_side/max(w,h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

PSM = "--oem 1 --psm 6"

def tesseract_ocr(pil_img:Image.Image)->str:
    txt = pytesseract.image_to_string(pil_img, lang="rus", config=PSM)
    txt = re.sub(r'[^0-9A-Za-zА-Яа-я.,:%() ]+', ' ', txt)
    return re.sub(r'\s{2,}', ' ', txt).strip()

def vision_correct(image_b64:str, raw_text:str)->str:
    system = "Ты OCR-эксперт. Исправь ошибки в тексте, сверяясь с изображением. Верни ОДНУ строку."
    user = [
        {"type":"input_text","text":f"Черновой OCR: «{raw_text}». Исправь так, чтобы буквы/цифры совпадали с изображением."},
        {"type":"input_image","image_url":f"data:image/jpeg;base64,{image_b64}"}
    ]
    msg=[{"role":"system","content":system},
         {"role":"user","content":user}]
    return gpt4o(msg, max_tokens=200)

def rule_mining(raw:str, fixed:str)->List[str]:
    system = ("Сравни два варианта OCR и выведи до 5 правил вида "
              "`s/ошибка/исправление/` или `delete(символ)`.")
    user = f"RAW: {raw}\nFIXED: {fixed}"
    msg=[{"role":"system","content":system},
         {"role":"user","content":user}]
    rules = gpt4o(msg, max_tokens=150, T=0.2)
    return [r.strip() for r in rules.splitlines() if r.strip()]

def apply_rules(text:str, rules:List[str])->str:
    for r in rules:
        if r.startswith("s/"):
            _,bad,good,_ = r.split("/",3)
            text = text.replace(bad,good)
        elif r.startswith("delete("):
            ch = r[7:-1]
            text = text.replace(ch,"")
    return text

def cer(a:str,b:str)->float:
    return Lev.distance(a,b)/max(1,len(b))

# ==============================================================
# 2.  Отбираем 15 случайных изображений
# ==============================================================
all_imgs = sorted(DATA.glob("Screenshot_*"))
SAMPLE = random.sample(all_imgs, 15)

# ==============================================================
# 3.  Проходим pipeline → собираем правила
# ==============================================================
results=[]; global_rules=[]
for p in tqdm(SAMPLE, desc="Processing"):
    pil = Image.open(p).convert("RGB")
    raw = tesseract_ocr(pil)
    jpeg_bytes = to_jpeg_bytes(pil)
    img64 = base64.b64encode(jpeg_bytes).decode()
    fixed = vision_correct(img64, raw)

    # локальные правила
    rules = rule_mining(raw, fixed)
    global_rules.extend(rules)

    results.append({
        "file":p.name,
        "raw":raw,
        "fixed":fixed,
        "rules":rules,
        "cer_raw":cer(raw,fixed),
        "cer_fixed":0.0
    })

# deduplicate rules
global_rules = list(dict.fromkeys(global_rules))
print("🎯 Сгенерировано правил:", len(global_rules))
print("\n".join(global_rules[:10]))

# ==============================================================
# 4.  Проверяем правила на новой подвыборке
# ==============================================================
TEST = [x for x in all_imgs if x not in SAMPLE][:10]
before=[]; after=[]
for p in TEST:
    pil = Image.open(p).convert("RGB")
    raw = tesseract_ocr(pil)
    corrected = apply_rules(raw, global_rules)
    before.append(cer(raw, corrected))     # расстояние raw vs corrected
    # считаем oracle CER через GPT-коррекцию
    fixed = vision_correct(
        base64.b64encode(to_jpeg_bytes(pil)).decode(), raw)
    after.append(cer(corrected, fixed))

print(f"\nДо правил  CER≈{np.mean(before):.2f} │ после правил  CER≈{np.mean(after):.2f}")

# ==============================================================
# 5.  Визуализируем 3 примера
# ==============================================================
plt.figure(figsize=(12,6))
for i,(p,bef) in enumerate(zip(TEST[:3], before),1):
    pil = Image.open(p).convert("RGB")
    raw = tesseract_ocr(pil)
    rule_txt = apply_rules(raw, global_rules)
    plt.subplot(3,2,2*i-1); plt.imshow(pil); plt.axis('off'); plt.title(p.name)
    txt = textwrap.fill(f"RAW: {raw}\nRULE→ {rule_txt}", 50)
    plt.subplot(3,2,2*i); plt.text(0,0.5,txt,fontsize=9); plt.axis('off')
plt.tight_layout(); plt.show()

# ==============================================================
# 6.  Сохраняем «чистую» разметку для дальнейшего обучения
# ==============================================================
clean_pairs = {r["file"]: r["fixed"] for r in results}
json.dump(clean_pairs, open("clean_labels.json","w",encoding="utf8"), ensure_ascii=False, indent=2)
print("💾 clean_labels.json сохранён.")
