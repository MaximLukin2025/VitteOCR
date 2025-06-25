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
