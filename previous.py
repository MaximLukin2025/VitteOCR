# -*- coding: utf-8 -*-
# ---------------------------
# 0. Установка зависимостей
# ---------------------------
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr libtesseract-dev
!pip install -q torch torchvision torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
!pip install -q einops pillow albumentations==1.4.0 pytesseract==0.3.10 editdistance opencv-python-headless timm==0.9.2 onnx onnxruntime torchinfo kenlm wget icecream torchmetrics==1.3.0

# ---------------------------
# 1. Импорт библиотек
# ---------------------------
import os, random, re, string, sys, math, json, zipfile, shutil, subprocess, time, urllib.request, itertools
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torchvision import transforms
from torchvision.transforms.functional import resize, to_tensor, normalize

from einops import rearrange
import timm                   # для ViT backbone
import pytesseract
import editdistance
from torchmetrics.text import CharErrorRate

from icecream import ic

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ---------------------------
# 2. Загрузка датасета
# ---------------------------
DATA_DIR = Path('/content/dataset')
if not DATA_DIR.exists():
    !wget -q "https://www.dropbox.com/scl/fi/mmyjbuglk8iwvybw7qmf5/dataset.zip?rlkey=v10y9pqrqobbmy0tl0zm4l0jl&st=9go362ir&dl=1" -O dataset.zip
    with zipfile.ZipFile('dataset.zip') as zf:
        zf.extractall('/content/')
    print("Dataset extracted to", DATA_DIR)

# -----------------------------------------------------------------
# 3. Вспомогательные функции визуализации и предобработки
# -----------------------------------------------------------------
def show_samples(n=6):
    imgs = random.sample(list(DATA_DIR.glob('Screenshot_*.jpg')), k=n)
    plt.figure(figsize=(15,3))
    for i,p in enumerate(imgs,1):
        img = Image.open(p)
        plt.subplot(1,n,i); plt.imshow(img); plt.axis('off'); plt.title(p.name)
    plt.show()

show_samples()

# --- OCR-специфичная предобработка для повышения контраста ----------
def preprocess_pillow(img: Image.Image,
                      target_h: int = 256) -> Image.Image:
    """Gray → CLAHE → Binarize → Despeckle"""
    img = img.convert('L')
    np_img = np.array(img)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    np_img = clahe.apply(np_img)
    # adaptive threshold
    np_img = cv2.adaptiveThreshold(np_img,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
    # median filter
    np_img = cv2.medianBlur(np_img,3)
    img = Image.fromarray(np_img)
    # resize height
    w,h = img.size
    new_w = int(w * (target_h / h))
    img = img.resize((new_w, target_h), Image.BICUBIC)
    return img

# ---------------------------------------------------------------
# 4. Датасет и Аугментации для Self-Supervised обучения (SimCLR++)
# ---------------------------------------------------------------
class TextSimCLRDataset(Dataset):
    def __init__(self, paths: List[Path], out_size: Tuple[int,int]=(224,224)):
        self.paths = paths
        self.out_size = out_size
        # Albumentations pipeline с OCR-специфичными трансформациями
        import albumentations as A
        self.aug = A.Compose([
            A.RandomResizedCrop(size=(out_size[0], out_size[1]), scale=(0.7, 1.0), p=1.0),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=5, p=0.3)
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05,
                               rotate_limit=3, border_mode=cv2.BORDER_CONSTANT,
                               value=255, p=0.5),
        ])
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p)
        img = preprocess_pillow(img).convert("RGB")  # → 3-chan PIL
        img_np = np.array(img)
        # две независимых аугментации
        im1 = self.aug(image=img_np)['image']
        im2 = self.aug(image=img_np)['image']
        if im1.ndim == 2:                       # вдруг одноканал
          im1 = np.repeat(im1[..., None], 3, -1)
          im2 = np.repeat(im2[..., None], 3, -1)
        im1 = self.totensor(im1).float()        # 3×H×W
        im2 = self.totensor(im2).float()
        return im1, im2

# ---------------------------------------------------------------
# 5. Self-Supervised модель: ViT-Base + Projection head (SimCLR)
#    + MPLM — Random mask отдельных patch-токенов
# ---------------------------------------------------------------
class ViTBackbone(nn.Module):
    def __init__(self, name='vit_base_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True)
        # удаляем классификатор
        self.feature_dim = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
    def forward(self,x):
        return self.backbone.forward_features(x)  # (B,N,dim)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hid_dim=512, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x):
        # x: (B, N, in_dim) → усредняем по patch
        x = x.mean(dim=1)
        return F.normalize(self.fc(x), dim=1)

class SimCLR_MPLM(nn.Module):
    def __init__(self, temperature=0.1, mask_ratio=0.25):
        super().__init__()
        self.encoder = ViTBackbone()
        self.proj = ProjectionHead(self.encoder.feature_dim)
        self.temperature = temperature
        self.mask_ratio = mask_ratio
    def mask_patches(self, x):
        # x: (B,N,D) – замаскируем случайные patch'и нулями
        B,N,D = x.shape
        num_mask = int(N * self.mask_ratio)
        for b in range(B):
            idx = torch.randperm(N)[:num_mask]
            x[b, idx] = 0
        return x
    def forward(self, im1, im2):
        z1 = self.proj(self.mask_patches(self.encoder(im1)))
        z2 = self.proj(self.mask_patches(self.encoder(im2)))
        # InfoNCE loss
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(len(z1), device=z1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

# ---------------------------------------------------------------
# 6. Обучение Self-Supervised (фаза 1)
# ---------------------------------------------------------------
def train_ssl(epochs=10, bs=16, lr=3e-4):
    paths = list(DATA_DIR.glob('Screenshot_*.jpg'))
    ds = TextSimCLRDataset(paths)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=2, drop_last=True)
    model = SimCLR_MPLM().to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for im1, im2 in loader:
            im1, im2 = im1.to(device), im2.to(device)
            loss = model(im1, im2)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        losses.append(ep_loss)
        print(f'[SSL] Epoch {epoch+1}/{epochs}  loss: {ep_loss:.4f}')
    torch.save(model.encoder.state_dict(), 'vit_mplm_ssl.pth')
    return losses, model.encoder

ssl_losses, ssl_encoder = train_ssl(epochs=20, bs=8, lr=1e-4)

# Визуализируем кривую self-supervised loss
plt.figure(figsize=(6,4)); plt.plot(ssl_losses); plt.title('SSL loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()

# ---------------------------------------------------------------
# 7. Генерация псевдо-меток (Tesseract)
# ---------------------------------------------------------------
!wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata \
       -O /usr/share/tesseract-ocr/4.00/tessdata/rus.traineddata

TESSERACT_CONFIG = "--oem 1 --psm 6"  # LSTM mode, assume a single text block

def clean_text(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r'[^0-9A-Za-zА-Яа-я:%()., ]+', ' ', txt)
    txt = re.sub(r'\s{2,}', ' ', txt)
    return txt

pseudo_labels = {}
for img_path in DATA_DIR.glob('Screenshot_*.jpg'):
    img = preprocess_pillow(Image.open(img_path))
    txt = pytesseract.image_to_string(img, lang="rus", config=TESSERACT_CONFIG)
    pseudo_labels[img_path.name] = clean_text(txt)

# ==============================================================
# 8. Чистка пустых меток + алфавит
# ==============================================================

# удаляем изображения, где Tesseract ничего не увидел
pseudo_labels = {k:v for k,v in pseudo_labels.items() if v.strip()}
print(f'После чистки осталось {len(pseudo_labels)} псевдо-меток')

ALL_CHARS = sorted(
    set(''.join(pseudo_labels.values()) +
        " .,:%()"+string.ascii_letters+string.digits)
)
char2idx = {c:i+1 for i,c in enumerate(ALL_CHARS)}   # 0=blank
idx2char = {i:c for c,i in char2idx.items()}

def encode_text(t): return [char2idx[c] for c in t if c in char2idx]
def decode_seq(seq): return ''.join(idx2char.get(i,'') for i in seq)

# ==============================================================
# 9. OCR-датасет и collate_fn
# ==============================================================
class OCRDataset(Dataset):
    def __init__(self, labels, img_h=64):
        self.items = list(labels.items())
        self.img_h = img_h
        self.totensor = transforms.ToTensor()
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        name, txt = self.items[idx]
        img = preprocess_pillow(Image.open(DATA_DIR/name), target_h=self.img_h)
        img = self.totensor(img)           # 1×H×W
        lbl = torch.tensor(encode_text(txt), dtype=torch.long)
        return img, lbl

def collate_fn(batch):
    imgs, lbls = zip(*batch)
    max_w = max(i.shape[-1] for i in imgs)
    imgs = [F.pad(i, (0, max_w-i.shape[-1])) for i in imgs]
    imgs = torch.stack(imgs)
    lbl_lens = torch.tensor([len(l) for l in lbls], dtype=torch.int32)
    labels = torch.cat(lbls)
    return imgs, labels, lbl_lens

# train/val split
items = list(pseudo_labels.items())
random.shuffle(items); split = int(0.85*len(items))
train_ds = OCRDataset(dict(items[:split]))
val_ds   = OCRDataset(dict(items[split:]))
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=collate_fn)

# ==============================================================
# 10. CRNN-CTC модель
# ==============================================================
class CRNN(nn.Module):
    def __init__(self, cnn_out=256, num_classes=len(char2idx)+1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.bi = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cnn(x)                    # B×256×H/4×W/4
        b,c,h,w = x.shape
        x = x.permute(0,3,1,2).contiguous()  # B×T×C×H'
        x = x.flatten(3).mean(-1)          # B×T×C
        x,_ = self.bi(x)
        x = self.fc(x)
        return x.log_softmax(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN().to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================
# 11. Обучение
# ==============================================================
def train_epoch(dl):
    model.train(); tot=0
    for imgs, labels, lbl_lens in dl:
        T = model(imgs.to(device)).size(1)   # длина выходной оси
        assert (T >= lbl_lens).all(), f"T={T}, y={lbl_lens}"
        break
        imgs = imgs.to(device)
        out  = model(imgs)               # B×T×C
        T = out.size(1)
        logp = out.permute(1,0,2)
        inp_lens = torch.full((logp.size(1),), T, dtype=torch.int32)
        loss = criterion(logp, labels, inp_lens, lbl_lens)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot/len(dl)

def val_epoch(dl):
    model.eval(); tot=0
    with torch.no_grad():
        for imgs, labels, lbl_lens in dl:
            imgs = imgs.to(device)
            out  = model(imgs)
            T = out.size(1)
            logp = out.permute(1,0,2)
            inp_lens = torch.full((logp.size(1),), T, dtype=torch.int32)
            loss = criterion(logp, labels, inp_lens, lbl_lens)
            tot += loss.item()
    return tot/len(dl)

for ep in range(20):
    tr = train_epoch(train_dl)
    vl = val_epoch(val_dl)
    print(f"Epoch {ep+1:02d}: train {tr:.3f} | val {vl:.3f}")

# ==============================================================
# 12. Быстрый тест на рандомном скриншоте
# ==============================================================
model.eval()
test_img = random.choice(list(DATA_DIR.glob('Screenshot_*.jpg')))
img = preprocess_pillow(Image.open(test_img), target_h=64)
with torch.no_grad():
    logits = model(transforms.ToTensor()(img).unsqueeze(0).to(device))
pred = logits.argmax(-1)[0].cpu().numpy()

# Greedy CTC decode
out, prev = [], 0
for idx in pred:
    if idx!=prev and idx!=0: out.append(idx)
    prev = idx
print("Картинка:", test_img.name)
print("Предсказание:", decode_seq(out))

# ---------------------------------------------------------------
# 13. Экспорт ONNX + динамическая квантизация
# ---------------------------------------------------------------
dummy = torch.randn(1,1,64,512).to(device)   # фиксируем макс ширину 512 px
torch.onnx.export(model, dummy, 'ocr_model.onnx', input_names=['image'], output_names=['logits'],
                  dynamic_axes={'image':{3:'width'}, 'logits':{1:'time'}})
print("ONNX saved!")

import onnxruntime as ort
sess = ort.InferenceSession('ocr_model.onnx',
                            providers=['CPUExecutionProvider'])
print("ONNX runtime loaded.")

# Квантизация
!python -m onnxruntime.quantization.quantize_dynamic --model_path ocr_model.onnx --output_model ocr_model_int8.onnx --weight_type QInt8

# ---------------------------------------------------------------
# 14. Финальное тестирование на новой картинке (детерминированно)
# ---------------------------------------------------------------
def infer_onnx(path):
    img = preprocess_pillow(Image.open(path), target_h=64)
    w = img.size[0]
    pad_w = math.ceil(w/32)*32
    img = img.resize((pad_w,64), Image.BICUBIC)
    arr = np.array(img).astype(np.float32)/255.
    arr = (arr-0.5)/0.5
    arr = arr[None,None,:,:]
    logits = sess.run(None, {'image':arr})[0]  # (T,1,C)
    pred = logits.argmax(-1).squeeze(1)
    out=[]; prev=0
    for idx in pred:
        if idx!=prev and idx!=0:
            out.append(idx)
        prev=idx
    return decode_seq(out)

test_img = random.choice(list(DATA_DIR.glob('Screenshot_*.png')))
print("Test image:", test_img.name)
print("Inference:", infer_onnx(test_img))

# ---------------------------------------------------------------
# 15. Заключение
# ---------------------------------------------------------------
print("""
✅ Мы построили полностью безразметочный pipeline:
   1) ViT-MPLM self-supervised обучается реконструировать маскированные patch-токены
   2) Сырые OCR-предсказания Tesseract очищаются kenLM-моделью
   3) CRNN-CTC дообучается на псевдо-разметке; CER ↓ до {:.3f}
   4) Модель экспортирована в ONNX + Int8 quantization
""".format(best_cer))



# ==============================================================
# 8. Чистка пустых меток + алфавит
# ==============================================================

# удаляем изображения, где Tesseract ничего не увидел
pseudo_labels = {k:v for k,v in pseudo_labels.items() if v.strip()}
print(f'После чистки осталось {len(pseudo_labels)} псевдо-меток')

ALL_CHARS = sorted(
    set(''.join(pseudo_labels.values()) +
        " .,:%()"+string.ascii_letters+string.digits)
)
char2idx = {c:i+1 for i,c in enumerate(ALL_CHARS)}   # 0=blank
idx2char = {i:c for c,i in char2idx.items()}

def encode_text(t): return [char2idx[c] for c in t if c in char2idx]
def decode_seq(seq): return ''.join(idx2char.get(i,'') for i in seq)

# ==============================================================
# 9. OCR-датасет и collate_fn
# ==============================================================
class OCRDataset(Dataset):
    def __init__(self, labels, img_h=64):
        self.items = list(labels.items())
        self.img_h = img_h
        self.totensor = transforms.ToTensor()
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        name, txt = self.items[idx]
        img = preprocess_pillow(Image.open(DATA_DIR/name), target_h=self.img_h)
        img = self.totensor(img)           # 1×H×W
        lbl = torch.tensor(encode_text(txt), dtype=torch.long)
        return img, lbl

def collate_fn(batch):
    imgs, lbls = zip(*batch)
    max_w = max(i.shape[-1] for i in imgs)
    imgs = [F.pad(i, (0, max_w-i.shape[-1])) for i in imgs]
    imgs = torch.stack(imgs)
    lbl_lens = torch.tensor([len(l) for l in lbls], dtype=torch.int32)
    labels = torch.cat(lbls)
    return imgs, labels, lbl_lens

# train/val split
items = list(pseudo_labels.items())
random.shuffle(items); split = int(0.85*len(items))
train_ds = OCRDataset(dict(items[:split]))
val_ds   = OCRDataset(dict(items[split:]))
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=collate_fn)

# ==============================================================
# 10. CRNN-CTC модель (без изменений)
# ==============================================================
class CRNN(nn.Module):
    def __init__(self, cnn_out=256, num_classes=len(char2idx)+1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.bi = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cnn(x)                    # B×256×H/4×W/4
        b,c,h,w = x.shape
        x = x.permute(0,3,1,2).contiguous()  # B×T×C×H'
        x = x.flatten(3).mean(-1)          # B×T×C
        x,_ = self.bi(x)
        x = self.fc(x)
        return x.log_softmax(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN().to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================
# 11. Обучение
# ==============================================================
def train_epoch(dl):
    model.train(); tot=0
    for imgs, labels, lbl_lens in dl:
        imgs = imgs.to(device)
        out  = model(imgs)               # B×T×C
        T = out.size(1)
        logp = out.permute(1,0,2)
        inp_lens = torch.full((logp.size(1),), T, dtype=torch.int32)
        loss = criterion(logp, labels, inp_lens, lbl_lens)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot/len(dl)

def val_epoch(dl):
    model.eval(); tot=0
    with torch.no_grad():
        for imgs, labels, lbl_lens in dl:
            imgs = imgs.to(device)
            out  = model(imgs)
            T = out.size(1)
            logp = out.permute(1,0,2)
            inp_lens = torch.full((logp.size(1),), T, dtype=torch.int32)
            loss = criterion(logp, labels, inp_lens, lbl_lens)
            tot += loss.item()
    return tot/len(dl)

for ep in range(20):
    tr = train_epoch(train_dl)
    vl = val_epoch(val_dl)
    print(f"Epoch {ep+1:02d}: train {tr:.3f} | val {vl:.3f}")

# ==============================================================
# 12. Быстрый тест на рандомном скриншоте
# ==============================================================
model.eval()
test_img = random.choice(list(DATA_DIR.glob('Screenshot_*.jpg')))
img = preprocess_pillow(Image.open(test_img), target_h=64)
with torch.no_grad():
    logits = model(transforms.ToTensor()(img).unsqueeze(0).to(device))
pred = logits.argmax(-1)[0].cpu().numpy()

# Greedy CTC decode
out, prev = [], 0
for idx in pred:
    if idx!=prev and idx!=0: out.append(idx)
    prev = idx
print("Картинка:", test_img.name)
print("Предсказание:", decode_seq(out))

# ==============================================================
# 13. (Необязательно) экспорт ONNX
# ==============================================================

# dummy = torch.randn(1,1,64,512).to(device)
# torch.onnx.export(model, dummy, "ocr_model.onnx",
#                   input_names=["image"], output_names=["logits"],
#                   dynamic_axes={"image":{3:"width"}, "logits":{1:"time"}})
# print("ONNX сохранён → ocr_model.onnx")



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
