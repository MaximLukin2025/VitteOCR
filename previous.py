# ==============================================================
#  0. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ==============================================================
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr libtesseract-dev
# русскую high-accuracy модель (~36 МБ)
!wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata \
       -O /usr/share/tesseract-ocr/4.00/tessdata/rus.traineddata

!pip install -q torch torchvision torchaudio==2.2.0--index-url https://download.pytorch.org/whl/cu118
!pip install -q albumentations==1.4.0 pillow pytesseract==0.3.10 editdistance \
               opencv-python-headless torchmetrics einops timm icecream

# ==============================================================
#  1. ИМПОРТ БИБЛИОТЕК
# ==============================================================
import os, random, re, string, math, json, zipfile, shutil, urllib.request, itertools, time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from einops import rearrange
import pytesseract
from torchmetrics.text import CharErrorRate
from icecream import ic

torch.manual_seed(17)
np.random.seed(17)
random.seed(17)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ Torch:", torch.__version__, "| Device:", device)

# ==============================================================
#  2. ЗАГРУЗКА ДАТАСЕТА (архив + распаковка)
# ==============================================================
DATA_URL = "https://www.dropbox.com/scl/fi/mmyjbuglk8iwvybw7qmf5/dataset.zip?rlkey=v10y9pqrqobbmy0tl0zm4l0jl&dl=1"
RAW_ZIP  = "/content/dataset.zip"
DATA_DIR = Path("/content/dataset")

if not DATA_DIR.exists():
    !wget -q "$DATA_URL" -O "$RAW_ZIP"
    with zipfile.ZipFile(RAW_ZIP) as zf:
        zf.extractall("/content/")
    # иногда архив содержит вложенную папку «dataset/dataset/...»
    if not list(DATA_DIR.glob("Screenshot_*")):
        inner = next(DATA_DIR.glob("dataset"), None)
        if inner:
            for f in inner.iterdir(): shutil.move(str(f), DATA_DIR)
            inner.rmdir()
    print("📦 Dataset unpacked →", DATA_DIR)

# ==============================================================
#  3. ВИЗУАЛЬНАЯ ПРОВЕРКА 6 СЛУЧАЙНЫХ КАРТИНОК
# ==============================================================
def show_samples(n=6):
    plt.figure(figsize=(15,3))
    for i,p in enumerate(random.sample(list(DATA_DIR.glob('Screenshot_*.*')), n),1):
        img = Image.open(p)
        plt.subplot(1,n,i); plt.imshow(img); plt.title(p.name); plt.axis('off')
    plt.show()
show_samples()

# ==============================================================
#  4. ПРЕДОБРАБОТКА ДЛЯ OCR (серый → CLAHE → адаптив. бинаризация)
# ==============================================================
def preprocess_pillow(img: Image.Image, target_h:int=96) -> Image.Image:
    img = img.convert('L')
    arr = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr  = clahe.apply(arr)
    arr  = cv2.adaptiveThreshold(arr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)
    arr  = cv2.medianBlur(arr,3)
    img  = Image.fromarray(arr)
    w,h  = img.size
    img  = img.resize((int(w*target_h/h), target_h), Image.BICUBIC)
    return img

# ==============================================================
#  5. ПРОВЕРКА TESSERACT НА 25 СЛУЧАЙНЫХ ФАЙЛАХ
# ==============================================================

TESS_CFG = "--oem 1 --psm 6"           # LSTM-only, «один блок текста»
def clean(txt:str) -> str:
    txt = re.sub(r'[^0-9A-Za-zА-Яа-я.,:%() ]+', ' ', txt)
    return re.sub(r'\s{2,}', ' ', txt).strip()

sample_paths = random.sample(list(DATA_DIR.glob("Screenshot_*.*")), 25)
lens = []
for p in sample_paths:
    txt = pytesseract.image_to_string(preprocess_pillow(Image.open(p)),
                                      lang="rus", config=TESS_CFG)
    txt = clean(txt)
    lens.append(len(txt))
    print(f"{p.name:>15} | {txt[:60]}")
print("Ср. длина предсказания Tesseract:", np.mean(lens))

# минимальная sanity-проверка: > 0.5 символа в среднем = ОК
assert np.mean(lens) > 0.5, "Tesseract ничего не видит – проверьте шрифт/язык!"

# ==============================================================
#  6. ГЕНЕРАЦИЯ ПСЕВДО-МЕТОК ДЛЯ ВСЕХ 200 ИЗОБРАЖЕНИЙ
# ==============================================================
pseudo = {}
for p in DATA_DIR.glob("Screenshot_*.*"):
    txt = pytesseract.image_to_string(preprocess_pillow(Image.open(p)),
                                      lang="rus", config=TESS_CFG)
    txt = clean(txt)
    if txt: pseudo[p.name] = txt
print(f"👓  {len(pseudo)}/{len(list(DATA_DIR.glob('Screenshot_*.*')))} "
      "изображений получили непустой текст")

# ==============================================================
#  7. АЛФАВИТ + ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================
EXTRA = "#/-+"
ALL_CHARS = sorted(set(''.join(pseudo.values()) + EXTRA +
                       string.ascii_letters + string.digits + ' .,:%()'))
char2idx = {c:i+1 for i,c in enumerate(ALL_CHARS)}    # 0 = blank
idx2char = {i:c for c,i in char2idx.items()}

def enc(t): return [char2idx[c] for c in t if c in char2idx]
def dec(seq): return ''.join(idx2char.get(i,'') for i in seq)

# ==============================================================
#  8. DATASET / DATALOADER
# ==============================================================
class OCRDataset(Dataset):
    def __init__(self, labels:Dict[str,str], img_h:int=96):
        self.items = list(labels.items())
        self.img_h = img_h
        self.tt = transforms.ToTensor()
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        name, txt = self.items[idx]
        img = preprocess_pillow(Image.open(DATA_DIR/name), self.img_h)
        return self.tt(img), torch.tensor(enc(txt), dtype=torch.long)

def ctc_collate(batch):
    # выбросим примеры, где текст длиннее, чем хронось (T)
    goods = []
    for img,lbl in batch:
        T = img.shape[-1]//4           # после двух MaxPool2d(2)
        if len(lbl) <= T: goods.append((img,lbl))
    if not goods: return None          # бросим весь батч
    imgs,lbls = zip(*goods)
    max_w = max(i.shape[-1] for i in imgs)
    imgs = [F.pad(i, (0,max_w-i.shape[-1])) for i in imgs]
    imgs = torch.stack(imgs)
    lbl_lens = torch.tensor([len(l) for l in lbls], dtype=torch.int32)
    labels   = torch.cat(lbls)
    return imgs, labels, lbl_lens

items = list(pseudo.items()); random.shuffle(items)
split = int(0.85*len(items))
train_ds = OCRDataset(dict(items[:split]))
val_ds   = OCRDataset(dict(items[split:]))
train_dl = DataLoader(train_ds, 8, True, collate_fn=ctc_collate)
val_dl   = DataLoader(val_ds,   8, False, collate_fn=ctc_collate)

# ==============================================================
#  9. CRNN-CTC МОДЕЛЬ (минимальная, но устойчивая)
# ==============================================================
class CRNN(nn.Module):
    def __init__(self, n_cls=len(char2idx)+1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU())
        self.rnn = nn.LSTM(256,256,2,bidirectional=True,batch_first=True)
        self.fc  = nn.Linear(512,n_cls)
    def forward(self,x):
        x = self.cnn(x)                       # B×256×H/4×W/4
        b,c,h,w = x.shape
        x = x.permute(0,3,1,2).reshape(b,w, c*h//1)   # B×T×C
        x,_ = self.rnn(x)
        return self.fc(x).log_softmax(-1)

model = CRNN().to(device)
crit  = nn.CTCLoss(blank=0, zero_infinity=True)
opt   = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================
# 10. ОБУЧЕНИЕ (60 эпох, вывод CER каждые 5)
# ==============================================================
def loop_epoch(dl, train=True):
    model.train() if train else model.eval()
    tot=0; metric = CharErrorRate()
    for batch in dl:
        if batch is None: continue
        imgs,lbls,lbl_lens = batch
        imgs = imgs.to(device)
        out  = model(imgs)     # B×T×C
        T    = torch.full((out.size(0),), out.size(1), dtype=torch.int32)
        loss = crit(out.permute(1,0,2), lbls, T, lbl_lens)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
        # CER
        pred = out.argmax(-1).cpu().numpy()
        gt_texts=[]; pr_texts=[]
        idx=0
        for i,l in enumerate(lbl_lens):
            gt_texts.append(dec(lbls[idx:idx+l].tolist()))
            idx+=l
            seq,prev = [],0
            for v in pred[i]:
                if v!=prev and v!=0: seq.append(v)
                prev=v
            pr_texts.append(dec(seq))
        metric.update(pr_texts, gt_texts)
    return tot/max(1,len(dl)), metric.compute().item()

for ep in range(1,61):
    tr, _   = loop_epoch(train_dl, True)
    if ep%5==0:
        vl, cer = loop_epoch(val_dl, False)
        print(f"Epoch {ep:02d} | train {tr:.3f} | val {vl:.3f} | CER {cer:.2%}")

# ==============================================================
# 11. ТЕСТ: вывод 5 примеров
# ==============================================================
model.eval()
plt.figure(figsize=(12,8))
for i,p in enumerate(random.sample(list(val_ds.items), 5),1):
    img = preprocess_pillow(Image.open(DATA_DIR/p[0]),96)
    with torch.no_grad():
        logits = model(transforms.ToTensor()(img).unsqueeze(0).to(device))
    seq,prev=[],0
    for v in logits.argmax(-1)[0].cpu().numpy():
        if v!=prev and v!=0: seq.append(v)
        prev=v
    txt = dec(seq)
    plt.subplot(5,2,2*i-1); plt.imshow(img,cmap='gray'); plt.axis('off'); plt.title(p[0])
    plt.subplot(5,2,2*i);   plt.text(0.01,0.5,txt,fontsize=10); plt.axis('off')
plt.show()

# ==============================================================
# 12. (ОПЦИОНАЛЬНО) ЭКСПОРТ ONNX
# ==============================================================
# dummy = torch.randn(1,1,96,512).to(device)
# torch.onnx.export(model, dummy, "ocr_model.onnx",
#                   input_names=["image"], output_names=["logits"],
#                   dynamic_axes={"image":{3:"width"}, "logits":{1:"time"}})
# print("📤 onnx-файл сохранён: ocr_model.onnx")
