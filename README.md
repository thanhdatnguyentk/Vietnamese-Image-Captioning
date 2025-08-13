# 🇻🇳 Vietnamese Image Captioning (UIT-ViIC × EfficientNet-B0 × BARTPho)

Tạo mô tả ảnh tự động **tiếng Việt** bằng kiến trúc **Encoder–Decoder**:  
**EfficientNet-B0** trích xuất đặc trưng ảnh, **BARTPho** sinh câu mô tả. Phù hợp cho tìm kiếm ảnh theo ngôn ngữ tự nhiên, trợ năng nội dung hình ảnh và xây dựng hệ thống AI đa phương thức.

---

## 📁 Cấu trúc dự án
```
Vietnamese-Image-Captioning/
├── download_vilc_uit_dataset.py   # Tải ảnh từ link trong annotations UIT-ViIC
├── image-caption.py                # Huấn luyện & đánh giá mô hình
├── metrics.py                      # Tính BLEU, ROUGE-L, METEOR, CIDEr, F1/Recall token-level
├── README.md
└── UIT-ViIC/                       # (tạo sau) dữ liệu
    ├── annotations/               # *.json: uitviic_captions_{train,val,test}2017.json
    ├── images/                    # train2017/, val2017/, test2017/ (tự tải bằng script)
    └── predictions/               # predicted_captions_bartPho.json
```

---

## 🔧 Yêu cầu
- Python ≥ 3.9
- PyTorch & TorchVision (GPU khuyến nghị)
- Thư viện: `transformers`, `evaluate`, `tqdm`, `Pillow`, `numpy`, `nltk`, `requests`

Cài nhanh (tham khảo):
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # chỉnh theo CUDA của bạn
pip install transformers evaluate tqdm pillow numpy nltk requests
```

> Lưu ý: một số metric (ví dụ METEOR) dùng qua `evaluate` có thể tải thêm dữ liệu NLTK ở lần chạy đầu.

---

## 📥 Chuẩn bị dữ liệu (UIT‑ViIC)
1) Đặt các file JSON annotations vào: `UIT-ViIC/annotations/`
   - `uitviic_captions_train2017.json`
   - `uitviic_captions_val2017.json`
   - `uitviic_captions_test2017.json`

2) Tải ảnh theo annotations bằng script:
```bash
python download_vilc_uit_dataset.py
```
> Mặc định script đang trỏ `val2017`. Bạn có thể sửa nhanh 2 hằng số đầu file:
> - `ANNOTATIONS_FILE` → tới JSON tương ứng (train/val/test)
> - `OUTPUT_IMAGE_DIR` → thư mục đích ảnh (train2017/val2017/test2017)

Thư mục kết quả:
```
UIT-ViIC/images/{train2017,val2017,test2017}/
```

---

## 🧠 Kiến trúc & pipeline
- **Encoder**: EfficientNet‑B0 (pretrained, TorchHub NVIDIA)
- **Decoder**: BARTPho (Seq2Seq cho tiếng Việt: `vinai/bartpho-syllable`)
- **Huấn luyện**: mixed precision (`torch.cuda.amp`), CrossEntropyLoss, Adam
- **Biến đổi ảnh**: Resize(256) → RandomCrop(224) → Normalize(Imagenet)
- **Thiết bị**: tự động chọn `cuda` nếu có, ngược lại `cpu`

---

## 🚀 Huấn luyện
Mở `image-caption.py` và chỉnh các cờ:
```python
RUN_TRAINING = True     # bật train
DEBUG_MODE = False      # đặt True để train/val nhanh với ít mẫu
```
Chạy:
```bash
python image-caption.py
```
Checkpoint tốt nhất sẽ lưu tại:
```
checkpoints/best_image_captioning_model_vietnamese.pth.tar
```

**Một số siêu tham số mặc định (có thể thay đổi trong file):**
- `EMBED_SIZE = 768` (tự khớp với d_model của BARTPho)
- `LR = 5e-5`, `NUM_EPOCHS = 30`, `BATCH_SIZE = 32`
- `CLIP_GRAD_NORM = 1.0`

---

## 🧪 Đánh giá
Để chỉ đánh giá trên **test** (không huấn luyện):
```python
RUN_TRAINING = False
LOAD_MODEL = True  # đảm bảo checkpoint đã tồn tại
```
Chạy:
```bash
python image-caption.py
```
Kết quả (dự đoán + một GT) sẽ được lưu tại:
```
UIT-ViIC/predictions/predicted_captions_bartPho.json
```

**Metric hỗ trợ:**
- BLEU, ROUGE‑L, METEOR, CIDEr
- F1 trung bình & Recall trung bình theo **token-level** (dựa trên tokenizer của BARTPho)

---

## 🖼️ Suy luận nhanh (ví dụ rời)
Ví dụ minimal suy luận cho **một ảnh** với checkpoint đã train (chạy trong môi trường dự án):

```python
import torch
from PIL import Image
from torchvision import transforms
# Đổi tên file `image-caption.py` thành `image_caption.py` để có thể import như 1 module.
from image_caption import ImageCaptioningModel, Vocabulary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vinai/bartpho-syllable"

# Khởi tạo vocab & model
vocab = Vocabulary(model_name=model_name)
EMBED_SIZE = vocab.tokenizer.model_max_length  # hoặc đặt 768 rồi đồng bộ trong model
model = ImageCaptioningModel(embed_size=768, bartpho_model_name=model_name, train_CNN=False, freeze_bartpho=False).to(DEVICE)

# Nạp checkpoint
ckpt = torch.load("checkpoints/best_image_captioning_model_vietnamese.pth.tar", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Tiền xử lý ảnh như lúc train
tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),  # dùng CenterCrop khi suy luận
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("path/to/your_image.jpg").convert("RGB")
img = tfm(img).to(DEVICE)

with torch.no_grad():
    caption = model.predict(img, vocab, max_length=50)
print(caption)
```

---

## 🧾 Troubleshooting
- **`PermissionError` khi lưu checkpoint** → kiểm tra quyền ghi/tạo thư mục `checkpoints/`.
- **`CUDA out of memory`** → giảm `BATCH_SIZE`, dùng ảnh 224x224, tắt `train_CNN` để freeze encoder.
- **Không tải được metric (BLEU/ROUGE/METEOR/CIDEr)** → đảm bảo đã cài `evaluate`; mạng Internet cần thiết lần đầu để tải metric; có thể cần `nltk` data.
- **Kết quả NaN/Inf** → script đã có kiểm tra NaN/Inf và bỏ batch lỗi; nên hạ `LR`, bật `clip_grad_norm`.

---

## 🙌 Ghi nhận
- **BARTPho**: VINAI (`vinai/bartpho-syllable` trên Hugging Face)
- **EfficientNet-B0**: NVIDIA TorchHub
- **UIT‑ViIC**: Bộ dữ liệu mô tả ảnh tiếng Việt

> Dự án dành cho mục đích học tập/nghiên cứu. Hãy tuân thủ giấy phép dữ liệu và mô hình tương ứng.

---

## 👤 Tác giả
**Nguyễn Thành Đạt** 

