# app_streamlit.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.cuda.amp import autocast
import io

# Đặt biến môi trường cho PyTorch CUDA Allocator để tránh phân mảnh bộ nhớ
os.environ["PYTORCH_CUDA_CONF"] = "expandable_segments:True"

# --- Cấu hình thiết bị (CPU/GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Cấu hình đường dẫn checkpoint ---
CHECKPOINT_PATH = "./checkpoints/best_image_captioning_model_vietnamese.pth.tar" 

# --- Tiền xử lý văn bản và xây dựng từ điển (Vocabulary) ---
class Vocabulary:
    """Tái tạo lớp Vocabulary từ mã gốc."""
    def __init__(self, model_name="vinai/bartpho-syllable", freq_threshold=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        self.pad_token = self.tokenizer.pad_token
        self.sos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_id = self.tokenizer.pad_token_id
        self.sos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return self.tokenizer.vocab_size

    def numericalize_caption(self, text):
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        return tokens

# --- Các lớp mô hình (Model Classes) ---
class EncoderCNN(nn.Module):
    """Tái tạo lớp EncoderCNN từ mã gốc."""
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet.classifier.fc = nn.Linear(self.efficientnet.classifier.fc.in_features, embed_size)
        self.train_CNN = train_CNN

    def forward(self, images):
        features = self.efficientnet(images)
        return features

class BartPhoDecoder(nn.Module):
    """Tái tạo lớp BartPhoDecoder từ mã gốc."""
    def __init__(self, bartpho_model_name="vinai/bartpho-syllable", embed_size=512, freeze_bartpho=False):
        super(BartPhoDecoder, self).__init__()
        self.bartpho = AutoModelForSeq2SeqLM.from_pretrained(bartpho_model_name).to(DEVICE)
        self.bart_hidden_size = self.bartpho.config.d_model
        
        self.feature_projector = nn.Linear(embed_size, self.bart_hidden_size)
        
    def forward(self, features, captions_input_seq):
        projected_features = self.feature_projector(features).unsqueeze(1)
        outputs = self.bartpho(
            decoder_input_ids=captions_input_seq,
            encoder_outputs=(projected_features,)
        )
        logits = outputs.logits 
        return logits

    def generate_caption(self, features, vocab, max_length=50):
        self.eval() 
        with torch.no_grad():
            projected_features = self.feature_projector(features).unsqueeze(1)
            encoder_attention_mask = torch.ones(projected_features.shape[:2], device=features.device)
            generated_ids = self.bartpho.generate(
                encoder_outputs=transformers.modeling_outputs.BaseModelOutput(last_hidden_state=projected_features),
                attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                eos_token_id=vocab.eos_id,
                pad_token_id=vocab.pad_id,
                bos_token_id=vocab.sos_id,
            )
            caption_text = vocab.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return caption_text

class ImageCaptioningModel(nn.Module):
    """Tái tạo lớp ImageCaptioningModel từ mã gốc."""
    def __init__(self, embed_size, bartpho_model_name="vinai/bartpho-syllable", train_CNN=False, freeze_bartpho=False):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN)
        self.decoder = BartPhoDecoder(bartpho_model_name, embed_size, freeze_bartpho=freeze_bartpho) 
        
    def forward(self, images, captions_input_seq):
        features = self.encoder(images)
        outputs = self.decoder(features, captions_input_seq)
        return outputs

    def predict(self, image, vocab, max_length=50):
        self.eval() 
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = self.encoder(image)
            caption_text = self.decoder.generate_caption(features, vocab, max_length) 
        self.train() 
        return caption_text

# --- Định nghĩa các tham số mô hình ---
EMBED_SIZE = 1024
MODEL_NAME = "vinai/bartpho-syllable"
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# --- Tải mô hình và từ điển chỉ một lần ---
@st.cache_resource
def load_model_and_vocab():
    """Tải mô hình và từ điển chỉ một lần."""
    vocab_obj = Vocabulary(model_name=MODEL_NAME)
    model_obj = ImageCaptioningModel(EMBED_SIZE, MODEL_NAME).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print("=> Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False) 
        # Cập nhật: Xử lý các khóa có tiền tố 'module.'
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_obj.load_state_dict(new_state_dict)
        model_obj.eval() # Đặt mô hình ở chế độ eval sau khi tải
        return model_obj, vocab_obj
    else:
        st.error(f"Lỗi: Không tìm thấy checkpoint tại '{CHECKPOINT_PATH}'. Vui lòng đảm bảo bạn đã huấn luyện và lưu mô hình.")
        st.stop()
    return None, None

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Image Captioning Tiếng Việt", layout="centered")

st.title("👨‍🔬 Ứng dụng Image Captioning Tiếng Việt")
st.markdown("### Sử dụng mô hình EfficientNet-B0 + BartPho-Syllable")

# Tải mô hình và từ điển bằng cache
model, vocab = load_model_and_vocab()

# Tạo một widget tải file
uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)

    # Nút dự đoán
    if st.button("Tạo chú thích"):
        with st.spinner("Đang dự đoán..."):
            # Tiền xử lý hình ảnh
            image_tensor = TRANSFORM(image).to(DEVICE)
            
            # Dự đoán chú thích
            try:
                predicted_caption = model.predict(image_tensor, vocab)
                st.success("Dự đoán hoàn tất!")
                st.markdown(f"**Chú thích được dự đoán:** `{predicted_caption}`")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")

else:
    st.info("Vui lòng tải lên một hình ảnh để tạo chú thích.")
