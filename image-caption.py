import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import json
import os
from collections import Counter
from tqdm import tqdm
import re
from torch.cuda.amp import autocast, GradScaler
import random
import matplotlib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers 
from metrics import evaluate_metrics 

os.environ["PYTORCH_CUDA_CONF"] = "expandable_segments:True"

# --- Cấu hình đường dẫn dữ liệu ---
DATA_DIR = './UIT-ViIC'
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images', 'train2017')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images', 'val2017')
IMAGE_DIR_TEST = os.path.join(DATA_DIR, 'images', 'test2017')

ANNOTATIONS_FILE_TRAIN = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_train2017.json')
ANNOTATIONS_FILE_VAL = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_val2017.json')
ANNOTATIONS_FILE_TEST = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_test2017.json')

PREDICTED_JSON_PATH = os.path.join(DATA_DIR, 'predictions', 'predicted_captions_bartPho.json')
# --- Thiết bị (CPU/GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Đọc tệp chú thích
def load_annotations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['annotations'], data['images']

# --- Tiền xử lý văn bản và xây dựng từ điển (Vocabulary) ---
class Vocabulary:
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

    def build_vocabulary(self, captions_list):
        counter = Counter()
        for caption in captions_list:
            tokens = self.tokenizer.tokenize(caption)
            counter.update(tokens)
        
        print(f"BartPho vocabulary size (from tokenizer): {len(self)}")
        print(f"Number of unique tokens in your dataset (after BartPho tokenization): {len(counter)}")

# --- Lớp Dataset cho UIT-ViIC ---
class ViICDataset(Dataset):
    def __init__(self, annotations, images_info, img_dir, vocab, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        
        self.image_id_to_filename = {img['id']: img['file_name'] for img in images_info}

        self.data = []
        for anno in annotations:
            img_id = anno['image_id']
            caption = str(anno['caption'])
            file_name = self.image_id_to_filename.get(img_id)
            if file_name:
                self.data.append((file_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, caption_text = self.data[idx]
        image_path = os.path.join(self.img_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        captions_token_ids = self.vocab.tokenizer.encode(
            caption_text,
            truncation=True,
            max_length=self.vocab.tokenizer.model_max_length,
            return_tensors="pt"
        ).squeeze(0)

        return image, captions_token_ids

# Hàm collate_fn để xử lý độ dài chú thích khác nhau (padding)
def collate_fn(batch, vocab):
    # Lọc bỏ các mục None nếu có (do lỗi trong __getitem__ hoặc lọc trước đó)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    images = [item[0] for item in batch]
    captions_token_ids = [item[1] for item in batch]

    images = torch.stack(images, dim=0)

    padded_captions = pad_sequence(captions_token_ids,
                                   batch_first=True,
                                   padding_value=vocab.pad_id)

    captions_input = padded_captions[:, :-1]
    captions_target = padded_captions[:, 1:]
    
    return images, captions_input, captions_target

# --- Model ---
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet.classifier.fc = nn.Linear(self.efficientnet.classifier.fc.in_features, embed_size)
        self.train_CNN = train_CNN

        if not train_CNN:
            print("Freezing EncoderCNN parameters.")
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        else:
            print("EncoderCNN parameters are trainable.")


    def forward(self, images):
        features = self.efficientnet(images)
        return features

class BartPhoDecoder(nn.Module):
    def __init__(self, bartpho_model_name="vinai/bartpho-syllable", embed_size=512, freeze_bartpho=False):
        super(BartPhoDecoder, self).__init__()
        
        self.bartpho = AutoModelForSeq2SeqLM.from_pretrained(bartpho_model_name).to(DEVICE)
        
        if freeze_bartpho:
            print(f"Freezing BartPho model parameters.")
            for param in self.bartpho.parameters():
                param.requires_grad = False
        
        self.bart_hidden_size = self.bartpho.config.d_model
        
        if embed_size != self.bart_hidden_size:
            print(f"Warning: EncoderCNN embed_size ({embed_size}) does not match BartPho hidden_size ({self.bart_hidden_size}).")
            print(f"Adjusting feature_projector to map {embed_size} to {self.bart_hidden_size}.")
            
        self.feature_projector = nn.Linear(embed_size, self.bart_hidden_size)
        
    def forward(self, features, captions_input_seq):
        projected_features = self.feature_projector(features).unsqueeze(1)
        
        if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
            print("Warning: NaN or Inf found in projected_features!")
            raise ValueError("NaN or Inf in projected_features")
        if torch.isnan(captions_input_seq).any() or torch.isinf(captions_input_seq).any():
            print("Warning: NaN or Inf found in captions_input_seq!")
            raise ValueError("NaN or Inf in captions_input_seq")

        outputs = self.bartpho(
            decoder_input_ids=captions_input_seq,
            encoder_outputs=(projected_features,)
        )
        
        logits = outputs.logits 
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN or Inf found in BartPho logits!")
            raise ValueError("NaN or Inf in BartPho logits")

        return logits

    def generate_caption(self, features, vocab, max_length=50):
        self.eval() 
        with torch.no_grad():
            projected_features = self.feature_projector(features).unsqueeze(1)

            encoder_attention_mask = torch.ones(projected_features.shape[:2], device=features.device)

            # Sử dụng model.generate()
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

# --- Hàm lưu/tải checkpoint ---
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    print("=> Loading checkpoint")
    # Đặt weights_only=False để tương thích với các checkpoint cũ và các đối tượng không phải trọng số
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False) 
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_cider_score = checkpoint.get('best_cider_score', -1.0)
    
    print(f"=> Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}, best CIDEr: {best_cider_score:.4f}")
    return start_epoch, best_cider_score

def train_model(model, train_loader, val_loader, optimizer, criterion, vocab, num_epochs,
                clip_grad=None, save_path="best_model.pth", load_path=None, start_epoch=0, best_cider_score=-1.0, scaler=None):
    
    model.to(DEVICE)
    
    if scaler is None:
        scaler = GradScaler()

    # Tải checkpoint nếu có đường dẫn
    if load_path and os.path.exists(load_path):
        start_epoch, best_cider_score = load_checkpoint(load_path, model, optimizer, scaler)
        print(f"Resuming training from epoch {start_epoch} with best CIDEr {best_cider_score:.4f}")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        model.train() 
        train_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        torch.cuda.empty_cache()

        for batch_idx, (images, captions_input, captions_target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            # Bỏ qua batch rỗng
            if images.numel() == 0:
                continue

            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"Warning: NaN or Inf found in images at batch {batch_idx}! Skipping batch.")
                continue
            if torch.isnan(captions_input).any() or torch.isinf(captions_input).any():
                print(f"Warning: NaN or Inf found in captions_input at batch {batch_idx}! Skipping batch.")
                continue
            if torch.isnan(captions_target).any() or torch.isinf(captions_target).any():
                print(f"Warning: NaN or Inf found in captions_target at batch {batch_idx}! Skipping batch.")
                continue
            
            images = images.to(DEVICE)
            captions_input = captions_input.to(DEVICE)
            captions_target = captions_target.to(DEVICE)

            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images, captions_input)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
            
            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            if np.isnan(current_loss) or np.isinf(current_loss):
                print(f"Warning: Loss is NaN/Inf at batch {batch_idx}! Stopping training for this epoch.")
                break
            
            train_loss += current_loss

        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"Epoch {epoch+1} Train Loss: nan (Training diverged)")
            print("Consider reducing learning rate or debugging data/model stability.")
            break
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval() 
        val_loss = 0.0
        all_predicted_captions = []
        all_ground_truth_captions = []

        with torch.no_grad():
            for batch_idx, (images, captions_input, captions_target) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")):
                if images.numel() == 0:
                    continue

                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"Warning: NaN or Inf found in val images at batch {batch_idx}! Skipping batch.")
                    continue
                if torch.isnan(captions_input).any() or torch.isinf(captions_input).any():
                    print(f"Warning: NaN or Inf found in val captions_input at batch {batch_idx}! Skipping batch.")
                    continue
                if torch.isnan(captions_target).any() or torch.isinf(captions_target).any():
                    print(f"Warning: NaN or Inf found in val captions_target at batch {batch_idx}! Skipping batch.")
                    continue

                images = images.to(DEVICE)
                captions_input = captions_input.to(DEVICE)
                captions_target = captions_target.to(DEVICE)

                with autocast():
                    outputs = model(images, captions_input)
                    loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
                
                current_val_loss = loss.item()
                if np.isnan(current_val_loss) or np.isinf(current_val_loss):
                    print(f"Warning: Val Loss is NaN/Inf at batch {batch_idx}! Skipping batch for metric calculation.")
                    continue

                val_loss += current_val_loss

                for i in range(images.size(0)):
                    single_image = images[i].unsqueeze(0)
                    predicted_text = model.predict(single_image, vocab) 
                    all_predicted_captions.append(predicted_text)

                    gt_token_ids = captions_target[i].cpu().tolist()
                    gt_text = vocab.tokenizer.decode([idx for idx in gt_token_ids if idx != vocab.pad_id], skip_special_tokens=True)
                    all_ground_truth_captions.append(gt_text)
        
        if len(val_loader) > 0 and (np.isnan(val_loss) or np.isinf(val_loss)):
            print(f"Epoch {epoch+1} Val Loss: nan (Validation diverged)")
            val_metrics = {'CIDEr': -1.0, 'F1_token_avg': 0.0, 'Recall_token_avg': 0.0} 
        else:
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

            if all_predicted_captions and all_ground_truth_captions:
                val_metrics = evaluate_metrics(all_predicted_captions, all_ground_truth_captions, vocab)
            else:
                val_metrics = {'CIDEr': -1.0, 'F1_token_avg': 0.0, 'Recall_token_avg': 0.0}
        
        print(f"Validation Metrics (Epoch {epoch+1}):")
        for metric_name, score in val_metrics.items():
            print(f"  {metric_name}: {score:.4f}")
        
        # Save checkpoint if current CIDEr is better
        if val_metrics['CIDEr'] > best_cider_score:
            best_cider_score = val_metrics['CIDEr']
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_cider_score': best_cider_score,
            }
            save_checkpoint(checkpoint_state, filename=save_path)
            print(f"Saved best model with CIDEr Score: {best_cider_score:.4f} at epoch {epoch+1}")
        
        if len(val_loader) > 0:
            dataiter = iter(val_loader)
            example_images, _, example_captions_target = next(dataiter) 
            
            example_image_tensor = example_images[0].to(DEVICE)
            
            predicted_caption_text = model.predict(example_image_tensor, vocab)
            
            ground_truth_caption_indices = example_captions_target[0].cpu().tolist()
            ground_truth_caption_text = vocab.tokenizer.decode(
                [idx for idx in ground_truth_caption_indices if idx != vocab.pad_id],
                skip_special_tokens=True
            )

            print(f"\nExample from Validation Set:")
            print(f"Ground Truth: {ground_truth_caption_text}")
            print(f"Predicted:    {predicted_caption_text}")


def evaluate_on_test_set(model, test_loader, vocab, test_annotations, test_images_info, transform, output_json_path="predictions.json"):
    """
    Evaluates the model on the test set and saves predictions to a JSON file.
    """
    model.to(DEVICE)
    model.eval() # Đặt model ở chế độ eval
    
    # 1. Thu thập tất cả ground truths theo image_id
    ground_truth_captions_by_image_id = {}
    for anno in test_annotations:
        img_id = str(anno['image_id']) # Đảm bảo key là string
        caption_text = str(anno['caption'])
        if img_id not in ground_truth_captions_by_image_id:
            ground_truth_captions_by_image_id[img_id] = []
        ground_truth_captions_by_image_id[img_id].append(caption_text)

    # 2. Tạo dự đoán cho mỗi ảnh duy nhất trong tập test
    predicted_captions_dict = {} # {image_id: predicted_caption_text}
    
    print("\n--- Generating predictions for Test Set ---")
    for img_info in tqdm(test_images_info, desc="Predicting on Test Images"):
        img_id = str(img_info['id']) # Đảm bảo key là string
        file_name = img_info['file_name']
        image_path = os.path.join(IMAGE_DIR_TEST, file_name)

        try:
            image = Image.open(image_path).convert("RGB")
            if transform:
                image = transform(image)
            
            image_tensor = image.to(DEVICE)
            predicted_caption = model.predict(image_tensor, vocab)
            predicted_captions_dict[img_id] = predicted_caption

        except Exception as e:
            print(f"Error predicting for image {file_name} (ID: {img_id}): {e}. Skipping this image for prediction.")
            continue

    # 3. Chuẩn bị dữ liệu cho việc tính toán metrics và lưu JSON
    final_predicted_for_metrics = {} # {image_id: predicted_text}
    final_ground_truths_for_metrics = {} # {image_id: [list of gt texts]}
    predictions_to_save = [] # List of dictionaries for JSON output

    # Lọc các image_id hợp lệ (có cả dự đoán và ground truth)
    valid_image_ids = set(predicted_captions_dict.keys()).intersection(set(ground_truth_captions_by_image_id.keys()))
    
    for img_id in valid_image_ids:
        predicted = predicted_captions_dict[img_id]
        ground_truths = ground_truth_captions_by_image_id[img_id] # Lấy tất cả GT cho ảnh này
        
        # Để tính metrics, chúng ta cần các dict tương ứng
        final_predicted_for_metrics[img_id] = predicted
        final_ground_truths_for_metrics[img_id] = ground_truths
        
        selected_ground_truth = ground_truths[0] if ground_truths else ""

        predictions_to_save.append({
            "id": int(img_id), 
            "caption": selected_ground_truth,
            "predict": predicted
        })

    # 4. Tính toán Metrics
    if final_predicted_for_metrics and final_ground_truths_for_metrics:
        test_metrics = evaluate_metrics(final_predicted_for_metrics, final_ground_truths_for_metrics, vocab)
        print("\nTest Set Metrics:")
        for metric_name, score in test_metrics.items():
            print(f"  {metric_name}: {score:.4f}")
    else:
        print("No valid predictions or ground truths to evaluate on test set.")

    # 5. Lưu kết quả dự đoán ra file JSON
    print(f"\nSaving predictions to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True) # Tạo thư mục nếu chưa có
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_to_save, f, ensure_ascii=False, indent=4)
    print("Predictions saved successfully.")

    # Optional: Display a few random predictions from test set
    print("\n--- Random Examples from Test Set Predictions ---")
    if len(predictions_to_save) > 0:
        sample_predictions = random.sample(predictions_to_save, min(5, len(predictions_to_save)))
        for entry in sample_predictions:
            print(f"Image ID: {entry['id']}")
            print(f"  Ground Truth (1 example): {entry['caption']}")
            print(f"  Predicted:                {entry['predict']}\n")
    else:
        print("No predictions to display.")



if __name__ == '__main__':
    DEBUG_MODE = False # Đặt False để sử dụng toàn bộ tập dữ liệu
    DEBUG_SAMPLES = 100 
    
    RUN_TRAINING = False # Đặt True để huấn luyện, False để chỉ đánh giá trên test set

    # --- Hyperparameters ---
    EMBED_SIZE = 768
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    LR = 0.00005
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    FREQ_THRESHOLD = 0
    CLIP_GRAD_NORM = 1.0

    # --- Checkpoint Configuration ---
    CHECKPOINT_DIR = "./checkpoints" # Thư mục để lưu checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    SAVE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_image_captioning_model_vietnamese.pth.tar")
    # LOAD_MODEL chỉ ảnh hưởng đến việc tiếp tục huấn luyện.
    LOAD_MODEL = True 
    LOAD_MODEL_PATH = SAVE_MODEL_PATH # Đường dẫn đến checkpoint bạn muốn tải

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_annotations, train_images_info = load_annotations(ANNOTATIONS_FILE_TRAIN)
    val_annotations, val_images_info = load_annotations(ANNOTATIONS_FILE_VAL)
    test_annotations, test_images_info = load_annotations(ANNOTATIONS_FILE_TEST)
    
    # THAY ĐỔI: Giới hạn số lượng mẫu nếu DEBUG_MODE = True
    if DEBUG_MODE and RUN_TRAINING: # Chỉ áp dụng DEBUG_MODE nếu đang trong chế độ huấn luyện
        print(f"DEBUG_MODE is ON. Using first {DEBUG_SAMPLES} samples for training and validation.")
        train_annotations = train_annotations[:DEBUG_SAMPLES]
        val_annotations = val_annotations[:DEBUG_SAMPLES]
        
        train_img_ids_in_subset = {anno['image_id'] for anno in train_annotations}
        val_img_ids_in_subset = {anno['image_id'] for anno in val_annotations}

        train_images_info = [img for img in train_images_info if img['id'] in train_img_ids_in_subset]
        val_images_info = [img for img in val_images_info if img['id'] in val_img_ids_in_subset]
    
    # Vocabulary cần được xây dựng trên toàn bộ tập train ban đầu để đảm bảo đầy đủ
    # Trừ khi bạn muốn debug vocabulary, thì cần tạo lại vocab với tập nhỏ
    all_train_captions = [anno['caption'] for anno in train_annotations]
    if DEBUG_MODE and not RUN_TRAINING: # Khi DEBUG_MODE nhưng không train, hãy dùng annotations gốc để build vocab
        original_train_annotations, _ = load_annotations(ANNOTATIONS_FILE_TRAIN)
        all_train_captions = [anno['caption'] for anno in original_train_annotations]

    model_name = "vinai/bartpho-syllable"
    vocab = Vocabulary(model_name=model_name)
    vocab.build_vocabulary(all_train_captions) 
    print(f"Kích thước từ điển (Vocabulary size): {len(vocab)}")
    sample_caption = "Người đàn ông đang đi bộ trên đường phố với một chiếc túi."
    numericalized_sample = vocab.numericalize_caption(sample_caption)
    print(f"Câu gốc: {sample_caption}")
    print(f"Dạng số hóa (không có special tokens): {numericalized_sample}")
    print(f"Giải mã: {vocab.tokenizer.decode(numericalized_sample, skip_special_tokens=True)}")

    train_dataset = ViICDataset(
        annotations=train_annotations,
        images_info=train_images_info,
        img_dir=IMAGE_DIR_TRAIN,
        vocab=vocab,
        transform=transform
    )

    val_dataset = ViICDataset(
        annotations=val_annotations,
        images_info=val_images_info,
        img_dir=IMAGE_DIR_VAL,
        vocab=vocab,
        transform=transform
    )
    test_dataset = ViICDataset(
        annotations=test_annotations,
        images_info=test_images_info,
        img_dir=IMAGE_DIR_TEST,
        vocab=vocab,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab)
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab)
    )
    print(f"\nSố lượng batch trong train_loader: {len(train_loader)}")
    print(f"Số lượng batch trong val_loader: {len(val_loader)}")
    print(f"Số lượng batch trong test_loader: {len(test_loader)}")
    
    print("\nKiểm tra một batch dữ liệu:")
    # Lấy một batch mẫu từ train_loader (nếu có dữ liệu)
    try:
        for images, captions_input, captions_target in train_loader:
            if images.numel() == 0:
                print("Skipped empty batch.")
                continue
            print(f"Kích thước tensor ảnh: {images.shape}")
            print(f"Kích thước tensor captions_input (decoder input): {captions_input.shape}")
            print(f"Kích thước tensor captions_target (labels): {captions_target.shape}")
            
            sample_caption_input_ids = captions_input[0].tolist()
            sample_caption_target_ids = captions_target[0].tolist()
            
            decoded_input_caption = vocab.tokenizer.decode(
                [idx for idx in sample_caption_input_ids if idx != vocab.pad_id], 
                skip_special_tokens=True
            )
            decoded_target_caption = vocab.tokenizer.decode(
                [idx for idx in sample_caption_target_ids if idx != vocab.pad_id], 
                skip_special_tokens=True
            )
            print(f"Chú thích input ví dụ (đã giải mã): {decoded_input_caption}")
            print(f"Chú thích target ví dụ (đã giải mã): {decoded_target_caption}")
            
            break # Chỉ kiểm tra batch đầu tiên
    except StopIteration:
        print("Train loader is empty, cannot display example batch.")

    temp_decoder = BartPhoDecoder(model_name)
    bart_hidden_size = temp_decoder.bart_hidden_size
    del temp_decoder
    
    if EMBED_SIZE != bart_hidden_size:
        print(f"WARNING: EMBED_SIZE ({EMBED_SIZE}) in main script does not match BartPho's d_model ({bart_hidden_size}).")
        print(f"Automatically setting EMBED_SIZE to BartPho's d_model for consistency.")
        EMBED_SIZE = bart_hidden_size

    model = ImageCaptioningModel(EMBED_SIZE, model_name, train_CNN=True, freeze_bartpho=False).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler() # Khởi tạo GradScaler ở đây để truyền vào train_model

    initial_start_epoch = 0
    initial_best_cider = -1.0

    if RUN_TRAINING:
        if LOAD_MODEL:
            if os.path.exists(LOAD_MODEL_PATH):
                initial_start_epoch, initial_best_cider = load_checkpoint(LOAD_MODEL_PATH, model, optimizer, scaler)
            else:
                print(f"Warning: Checkpoint '{LOAD_MODEL_PATH}' not found. Starting training from scratch.")

        train_model(model, train_loader, val_loader, optimizer, criterion, vocab, NUM_EPOCHS, 
                    CLIP_GRAD_NORM, SAVE_MODEL_PATH, 
                    load_path=LOAD_MODEL_PATH if LOAD_MODEL else None,
                    start_epoch=initial_start_epoch, 
                    best_cider_score=initial_best_cider,
                    scaler=scaler)

        print("\n--- Training Finished ---")

    # Always load the best model (if exists) for evaluation, regardless of RUN_TRAINING
    if os.path.exists(SAVE_MODEL_PATH):
        print(f"\nLoading best model from {SAVE_MODEL_PATH} for evaluation.")
        # Load chỉ model state_dict khi đánh giá, không cần optimizer hay scaler
        checkpoint = torch.load(SAVE_MODEL_PATH, map_location=DEVICE, weights_only=False) 
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"\nWarning: Best model not found at {SAVE_MODEL_PATH}. Cannot perform evaluation on test set.")
        exit() # Thoát nếu không tìm thấy mô hình để đánh giá

    # Chạy đánh giá trên tập test
    evaluate_on_test_set(model, test_loader, vocab, test_annotations, test_images_info, transform, output_json_path=PREDICTED_JSON_PATH)    