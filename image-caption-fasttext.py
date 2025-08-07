import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import json
import os
from collections import Counter
from tqdm import tqdm
import re
from underthesea import word_tokenize
import random 
import matplotlib.pyplot as plt 
import numpy as np 
# --- Cấu hình đường dẫn dữ liệu ---
DATA_DIR = './UIT-ViIC' # Thay thế bằng đường dẫn thực tế của bạn
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images', 'train2017')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images', 'val2017')
IMAGE_DIR_TEST = os.path.join(DATA_DIR, 'images', 'test2017')

ANNOTATIONS_FILE_TRAIN = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_train2017.json')
ANNOTATIONS_FILE_VAL = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_val2017.json')
ANNOTATIONS_FILE_TEST = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_test2017.json')

# --- Thiết bị (CPU/GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Đọc tệp chú thích
def load_annotations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['annotations'], data['images']

# --- Tiền xử lý văn bản và xây dựng từ điển (Vocabulary) ---
# Đường dẫn tới file FastText đã giải nén
FASTTEXT_PATH = './cc.vi.300.vec' # Đảm bảo thay đổi đường dẫn này

def load_fasttext_embeddings(filepath, vocab, embed_dim):
    """
    Tải các vector nhúng FastText và tạo ma trận embedding cho từ điển của bạn.

    Args:
        filepath (str): Đường dẫn đến file .vec của FastText.
        vocab (Vocabulary): Đối tượng từ điển (Vocabulary) của bạn.
        embed_dim (int): Kích thước chiều của vector nhúng (phải khớp với FastText, ví dụ: 300).

    Returns:
        torch.Tensor: Ma trận embedding có kích thước (vocab_size, embed_dim).
    """
    
    # Khởi tạo ma trận embedding với các giá trị ngẫu nhiên
    # Điều này để đảm bảo các từ không tìm thấy trong FastText vẫn có vector
    # Hoặc các token đặc biệt (<PAD>, <SOS>, <EOS>, <UNK>) sẽ được xử lý riêng
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))

    # Đánh dấu các vector đã được điền từ FastText
    found_words = 0

    print(f"Loading FastText embeddings from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        # Dòng đầu tiên trong file .vec thường chứa số lượng từ và kích thước chiều
        # VD: 2000000 300
        header = f.readline().split()
        total_fasttext_words = int(header[0])
        fasttext_embed_dim = int(header[1])

        if fasttext_embed_dim != embed_dim:
            print(f"Warning: FastText embedding dimension ({fasttext_embed_dim}) does not match specified embed_dim ({embed_dim}). Using FastText's dimension.")
            embed_dim = fasttext_embed_dim
            # Cần tạo lại embedding_matrix nếu embed_dim thay đổi
            embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))


        for line in tqdm(f, total=total_fasttext_words, desc="Processing FastText vectors"):
            parts = line.strip().split(' ')
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)

            if word in vocab.stoi:
                idx = vocab.stoi[word]
                embedding_matrix[idx] = vector
                found_words += 1
    
    print(f"Found {found_words} words from FastText in your vocabulary ({found_words/len(vocab)*100:.2f}%).")
    
    # Xử lý các token đặc biệt nếu cần (tùy chỉnh theo yêu cầu)
    # Ví dụ: đặt vector cho <PAD> là 0
    if "<PAD>" in vocab.stoi:
        embedding_matrix[vocab.stoi["<PAD>"]] = np.zeros(embed_dim)

    return torch.tensor(embedding_matrix, dtype=torch.float32)
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3} # Lưu ý sửa lỗi chính tả ở đây nếu có
        self.freq_threshold = freq_threshold
        self.cleaned_captions = []

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, captions):
        all_words = []
        for caption in tqdm(captions, desc="Cleaning and tokenizing captions"):
            # Vẫn giữ nguyên làm sạch và tách từ, nhưng hãy kiểm tra lại như đã thảo luận trước
            # Vấn đề về 'ngi', 'b', 'vi mt' cần được xử lý ở đây.
            # Dưới đây là một gợi ý để làm sạch tốt hơn, tập trung vào việc chỉ giữ các ký tự chữ cái và số
            # và sau đó dùng word_tokenize
            cleaned_caption = re.sub(r'\s+', ' ', caption).strip().lower() # Chỉ xử lý dấu cách và chuyển lower
            tokens = word_tokenize(cleaned_caption, format="word")

            
            self.cleaned_captions.append(tokens)
            all_words.extend(tokens)

        word_counts = Counter(all_words)
        idx = len(self.itos)

        for word, count in word_counts.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize_caption(self, caption):
        cleaned_caption = re.sub(r'[^a-zA-Z0\s]', ' ', caption)
        cleaned_caption = re.sub(r'\s+', ' ', cleaned_caption).strip().lower()
        tokens = word_tokenize(cleaned_caption, format="word")
        
        numericalized_tokens = [self.stoi["<SOS>"]]
        for token in tokens:
            numericalized_tokens.append(self.stoi.get(token, self.stoi["<UNK>"]))
        numericalized_tokens.append(self.stoi["<EOS>"])
        return numericalized_tokens

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
            caption = anno['caption']
            file_name = self.image_id_to_filename.get(img_id)
            if file_name:
                self.data.append((file_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, caption = self.data[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        numericalized_caption = self.vocab.numericalize_caption(caption)
        caption_tensor = torch.tensor(numericalized_caption)
        
        return image, caption_tensor

# Hàm collate_fn để xử lý độ dài chú thích khác nhau (padding)
def collate_fn(batch):
    images = []
    captions = []
    for img, cap in batch:
        images.append(img)
        captions.append(cap)
    
    images = torch.stack(images, 0)
    
    max_len = max(len(c) for c in captions)
    # print(f"collate_fn: Current batch_size={len(batch)}, max_len={max_len}") # keep this
    padded_captions = torch.zeros((len(captions), max_len), dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
        
    return images, padded_captions
# Model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer (classifier)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # Add a fully connected layer to project features to desired embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Thêm Dropout để chống overfitting

    def forward(self, images):
        # (batch_size, 3, H, W) -> (batch_size, 512, 1, 1) if ResNet18
        features = self.resnet(images) 
        # (batch_size, 512, 1, 1) -> (batch_size, 512)
        features = features.view(features.size(0), -1) 
        # (batch_size, 512) -> (batch_size, embed_size)
        features = self.dropout(self.relu(self.fc(features)))
        return features
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # W_h * h_e (encoder_features)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # W_s * s_t (decoder_hidden_state)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # V_att * tanh(...)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # softmax trên các đặc trưng của ảnh

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, encoder_dim) (trong trường hợp EncoderCNN output là 1 vector)
        # decoder_hidden: (batch_size, decoder_dim) (hidden state của LSTM)

        # Vì EncoderCNN của chúng ta xuất ra một vector duy nhất cho toàn bộ ảnh,
        # attention sẽ được tính trên chính vector đặc trưng này.
        # Nếu EncoderCNN xuất ra feature map (ví dụ: từ conv layer cuối),
        # thì encoder_out sẽ là (batch_size, num_pixels, encoder_dim)
        # và attention sẽ học cách "phân bổ" sự chú ý trên các "pixels".
        # Với EncoderCNN hiện tại, nó hơi đơn giản hóa, nhưng vẫn minh họa được cơ chế.
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden) # (batch_size, attention_dim)
        
        # Additive attention
        # (batch_size, attention_dim) -> (batch_size, 1)
        energy = self.full_att(self.relu(att1 + att2)) 
        
        # Softmax để có trọng số attention
        # (batch_size, 1) -> (batch_size, 1)
        alpha = self.softmax(energy) 

        # Tính context vector
        # (batch_size, 1) * (batch_size, encoder_dim) = (batch_size, encoder_dim)
        context = (encoder_out * alpha).squeeze(1) 
        
        return context, alpha # context vector và trọng số attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embedding_weights=None):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Kiểm tra nếu embedding_weights được cung cấp
        if embedding_weights is not None:
            # Khởi tạo embedding layer với pre-trained weights
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
            # freeze=False (mặc định) cho phép các weights này được cập nhật trong quá trình huấn luyện.
            # Đặt freeze=True nếu bạn muốn giữ chúng cố định.
        else:
            # Khởi tạo ngẫu nhiên nếu không có pre-trained weights
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
        self.attention = Attention(encoder_dim=embed_size, decoder_dim=hidden_size, attention_dim=hidden_size)
        
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)


    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, max_seq_len) - This is captions_input from train_model, which is captions[:, :-1]
        
        # --- IMPORTANT CHANGE HERE ---
        # captions_for_embedding will be captions_input, which excludes <EOS>
        captions_for_embedding = captions[:, :-1]
        # The loop should run for the length of captions_input
        # Let's rename 'captions' to 'captions_input_seq' for clarity in this function
        captions_input_seq = captions # This 'captions' argument IS captions_input from train_model
        
        embeddings = self.embedding(captions_input_seq) # (batch_size, seq_len_input, embed_size)
        
        # Khởi tạo h và c
        h = self.init_h(features) 
        c = self.init_c(features)
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1) 
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1) 

        predictions = []
        

        expected_iterations = captions_input_seq.size(1) 

        
        for t in range(expected_iterations): # Loop for each token in captions_input_seq
            word_embedding = embeddings[:, t, :] # (batch_size, embed_size)

            current_h_state = h.squeeze(0) if self.num_layers == 1 else h[-1]
            context, _ = self.attention(features, current_h_state)
            
            lstm_input = torch.cat((word_embedding, context), dim=1) 
            lstm_input = lstm_input.unsqueeze(1) # (batch_size, 1, embed_size * 2)


            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(lstm_output.squeeze(1))) 
            predictions.append(output)
        return torch.stack(predictions, dim=1)
        
    def generate_caption(self, features, vocab, max_length=50):
        # features: (1, embed_size) - Batch size là 1 cho inference
        
        # Khởi tạo hidden và cell state
        h = self.init_h(features) 
        c = self.init_c(features)
        h = h.unsqueeze(0) 
        c = c.unsqueeze(0)

        caption_sequence = []
        # Bắt đầu với token <SOS>
        # (1, 1)
        word = torch.tensor(vocab.stoi["<SOS>"]).view(1, 1).to(DEVICE) 

        for _ in range(max_length):
            # Lấy embedding của từ hiện tại
            # (1, 1, embed_size)
            word_embedding = self.embedding(word) 

            # Tính toán context vector
            # (1, hidden_size)
            current_h_state = h.squeeze(0) 
            context, _ = self.attention(features, current_h_state)
            
            # Nối word_embedding và context vector
            # (1, embed_size * 2)
            lstm_input = torch.cat((word_embedding.squeeze(1), context), dim=1) 
            
            # (1, 1, embed_size * 2)
            lstm_input = lstm_input.unsqueeze(1) 

            # Truyền qua LSTM
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Dự đoán từ
            # (1, vocab_size)
            output = self.fc(lstm_output.squeeze(1)) 
            
            # Chọn từ có xác suất cao nhất
            # (1)
            predicted_word_idx = output.argmax(dim=1) 
            
            caption_sequence.append(predicted_word_idx.item())

            # Nếu dự đoán là <EOS>, dừng lại
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Cập nhật từ tiếp theo
            word = predicted_word_idx.unsqueeze(0) # (1,1)

        # Giải mã chuỗi số thành văn bản
        decoded_caption = [vocab.itos[idx] for idx in caption_sequence]
        return decoded_caption

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embedding_weights=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, embedding_weights=embedding_weights)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def predict(self, image, vocab, max_length=50):
        # Hàm predict sẽ không thay đổi
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(DEVICE))
            # features.squeeze(0) nếu bạn muốn đảm bảo nó là (embed_size,)
            caption = self.decoder.generate_caption(features.squeeze(0), vocab, max_length) 
        self.train()
        return caption

def train_model(model, train_loader, val_loader, optimizer, criterion, vocab, num_epochs,
                clip_grad=None, save_path="best_model.pth"):
    
    model.to(DEVICE)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Chuyển sang chế độ huấn luyện
        train_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, captions) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images = images.to(DEVICE)
            captions_input = captions[:, :-1].to(DEVICE)
            captions_target = captions[:, 1:].to(DEVICE)

            optimizer.zero_grad()
            
            # Call the model
            outputs = model(images, captions_input) # This is where DecoderRNN.forward is called
            
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
            
            loss.backward()

            # Optional: Gradient clipping để tránh vanishing/exploding gradients
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval() # Chuyển sang chế độ đánh giá
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, captions) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")):
                images = images.to(DEVICE)
                captions_input = captions[:, :-1].to(DEVICE)
                captions_target = captions[:, 1:].to(DEVICE)

                outputs = model(images, captions_input)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # Lưu mô hình nếu val_loss tốt hơn
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {best_loss:.4f}")
        
        # In một ví dụ dự đoán từ tập validation
        # Lấy ngẫu nhiên một ảnh từ val_loader
        if len(val_loader) > 0:
            random_batch_idx = random.randint(0, len(val_loader) - 1)
            example_images, example_captions = next(iter(val_loader)) # Lấy lại iterator để có thể lấy batch ngẫu nhiên
            example_image = example_images[0].cpu() # Lấy ảnh đầu tiên trong batch
            
            predicted_caption_tokens = model.predict(example_image, vocab)
            predicted_caption = ' '.join([token for token in predicted_caption_tokens if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]]) # Loại bỏ token đặc biệt
            
            # Giải mã chú thích gốc để so sánh
            ground_truth_caption_indices = example_captions[0].cpu().tolist()
            ground_truth_caption_tokens = [vocab.itos[idx] for idx in ground_truth_caption_indices]
            ground_truth_caption = ' '.join([token for token in ground_truth_caption_tokens if token not in ["<SOS>", "<EOS>", "<PAD>"]])

            print(f"\nExample from Validation Set:")
            print(f"Ground Truth: {ground_truth_caption}")
            print(f"Predicted:    {predicted_caption}")

            # Hiển thị ảnh (tùy chọn)
            # plt.imshow(example_image.permute(1, 2, 0)) # Chuyển tensor (C, H, W) về (H, W, C)
            # plt.title(f"Predicted: {predicted_caption}")
            # plt.show()
# --- Đảm bảo rằng tất cả code khởi tạo DataLoader đều nằm trong if __name__ == '__main__': ---
if __name__ == '__main__':
        # --- Hyperparameters ---
    EMBED_SIZE = 256        # Kích thước của word embeddings và feature ảnh
    HIDDEN_SIZE = 256       # Kích thước của hidden state LSTM
    NUM_LAYERS = 1          # Số lớp LSTM
    LR = 0.001              # Learning Rate
    NUM_EPOCHS = 10         # Số lượng epoch huấn luyện
    BATCH_SIZE = 32         
    NUM_WORKERS = 0         # Đặt 0 nếu gặp lỗi multiprocessing trên Windows, sau đó tăng dần
    FREQ_THRESHOLD = 5      # Ngưỡng tần suất từ cho Vocabulary
    CLIP_GRAD_NORM = 1.0    # Gradient clipping để ổn định huấn luyện
    # Định nghĩa biến đổi ảnh (Image Transformations)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Tải chú thích
    train_annotations, train_images_info = load_annotations(ANNOTATIONS_FILE_TRAIN)
    val_annotations, val_images_info = load_annotations(ANNOTATIONS_FILE_VAL)
    test_annotations, test_images_info = load_annotations(ANNOTATIONS_FILE_TEST)
    
    # Tạo từ điển từ tập huấn luyện
    all_train_captions = [anno['caption'] for anno in train_annotations]
    vocab = Vocabulary(freq_threshold=0) 
    vocab.build_vocabulary(all_train_captions)

    print(f"Kích thước từ điển (Vocabulary size): {len(vocab)}")
    sample_caption = "Người đàn ông đang đi bộ trên đường phố với một chiếc túi."
    numericalized_sample = vocab.numericalize_caption(sample_caption)
    print(f"Câu gốc: {sample_caption}")
    print(f"Dạng số hóa: {numericalized_sample}")
    print(f"Giải mã: {[vocab.itos[i] for i in numericalized_sample]}")
    # --- Tải và xử lý FastText Embeddings ---
    # Đảm bảo FASTTEXT_PATH được cấu hình đúng
    if not os.path.exists(FASTTEXT_PATH):
        print(f"Error: FastText embedding file not found at {FASTTEXT_PATH}")
        print("Please download 'cc.vi.300.vec.gz' from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz and extract it.")
        print("Then update the FASTTEXT_PATH variable.")
        # Thoát hoặc sử dụng khởi tạo embedding ngẫu nhiên nếu không tìm thấy file
        embedding_weights = None 
    else:
        embedding_weights = load_fasttext_embeddings(FASTTEXT_PATH, vocab, EMBED_SIZE)
        print(f"Kích thước ma trận embedding đã tải: {embedding_weights.shape}")

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
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"\nSố lượng batch trong train_loader: {len(train_loader)}")
    print(f"Số lượng batch trong val_loader: {len(val_loader)}")
    print(f"Số lượng batch trong test_loader: {len(test_loader)}")
    # Kiểm tra một batch dữ liệu
    print("\nKiểm tra một batch dữ liệu:")
    for images, captions in train_loader:
        print(f"Kích thước tensor ảnh: {images.shape}")
        print(f"Kích thước tensor chú thích: {captions.shape}")
        
        sample_caption_indices = captions[0].tolist()
        decoded_caption = []
        for idx in sample_caption_indices:
            if idx == vocab.stoi["<SOS>"] or idx == vocab.stoi["<EOS>"] or idx == vocab.stoi["<PAD>"]:
                continue
            decoded_caption.append(vocab.itos[idx])
        print(f"Chú thích ví dụ (đã giải mã): {' '.join(decoded_caption)}")
        
        break
        # --- Khởi tạo Model, Optimizer, Loss Function ---
    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS, embedding_weights=embedding_weights).to(DEVICE)
    model.decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS, embedding_weights=embedding_weights).to(DEVICE)

    # CrossEntropyLoss bỏ qua PAD token
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"]) 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Huấn luyện mô hình ---
    train_model(model, train_loader, val_loader, optimizer, criterion, vocab, NUM_EPOCHS, CLIP_GRAD_NORM, "image_captioning_model_vietnamese_pretrained_embeddings.pth")
    print("\n--- Training Finished ---")

# --- Dự đoán trên tập kiểm tra ---
    model.load_state_dict(torch.load("image_captioning_model_vietnamese_pretrained_embeddings.pth"))
    model.eval()

    print("\n--- Predicting on Test Set ---")
    with torch.no_grad():
        for images, captions in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            predicted_captions = []
            for i in range(images.size(0)):
                caption = model.predict(images[i], vocab)
                predicted_captions.append(' '.join(caption))
            
            # Hiển thị kết quả dự đoán
            for i, caption in enumerate(predicted_captions):
                print(f"Image {i+1}: {caption}")
            break  # Chỉ hiển thị một batch để kiểm tra