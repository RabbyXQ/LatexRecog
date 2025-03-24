import kagglehub
import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# --- Data Loading and Tokenization ---
path = kagglehub.dataset_download("gregoryeritsyan/im2latex-230k")
dataset_path = os.path.join(path, "PRINTED_TEX_230k")
json_file = os.path.join(dataset_path, "230k.json")
corr_images_file = os.path.join(dataset_path, "corresponding_png_images.txt")
formulas_file = os.path.join(dataset_path, "final_png_formulas.txt")
images_dir = os.path.join(dataset_path, "generated_png_images")

with open(json_file, 'r') as f:
    vocab = json.load(f)

vocab = {k: int(v) for k, v in vocab.items()}
if '<PAD>' not in vocab:
    vocab['<PAD>'] = len(vocab)
if '<UNK>' not in vocab:
    vocab['<UNK>'] = len(vocab)
if '<E>' not in vocab:
    vocab['<E>'] = len(vocab)

char_to_idx = vocab
idx_to_char = {idx: char for char, idx in vocab.items()}
vocab_size = len(vocab)

with open(corr_images_file, 'r') as f:
    corr_images = [line.strip() for line in f.readlines()]
with open(formulas_file, 'r') as f:
    formulas = [line.strip() for line in f.readlines()]

image_formula_pairs = list(zip(corr_images, formulas))

# --- Dataset Class ---
class Im2LatexDataset(Dataset):
    def __init__(self, pairs, images_dir, char_to_idx, transform=None, max_len=1024):
        self.pairs = pairs
        self.images_dir = images_dir
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, formula = self.pairs[idx]
        image_path = os.path.join(self.images_dir, image_name)
        img = Image.open(image_path).convert('L')
        if self.transform:
            img = self.transform(img)

        formula_idx = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in formula]
        # Reserve space for <E> token
        if len(formula_idx) >= self.max_len - 1:
            formula_idx = formula_idx[:self.max_len - 1]
        formula_idx.append(self.char_to_idx['<E>'])

        formula_len = len(formula_idx)
        # Pad the rest
        formula_idx += [self.char_to_idx['<PAD>']] * (self.max_len - formula_len)

        return img, torch.tensor(formula_idx, dtype=torch.long), torch.tensor(formula_len, dtype=torch.long)

# --- Custom collate_fn ---
def custom_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)
    target_lengths = torch.stack(target_lengths, 0)
    return images, targets, target_lengths

# --- Model Definition ---
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.rnn = nn.LSTM(512 * 8, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(batch, width, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# --- Training Setup ---
transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor()
])

train_pairs = image_formula_pairs[:1000]
dataset = Im2LatexDataset(train_pairs, images_dir, char_to_idx, transform=transform, max_len=120)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss(blank=char_to_idx['<PAD>'], reduction='mean')

# --- Training Loop ---
def train_model(model, dataloader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)

            loss = ctc_loss(outputs.transpose(0, 1), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(dataloader):.4f}")

# --- Train the Model ---
print("Starting training...")
train_model(model, dataloader, epochs=5)

# --- Save the Model ---
torch.save(model.state_dict(), "latex_recog_model.pth")
print("Model saved to latex_recog_model.pth")

# --- Inference ---
def predict_latex(image_path, model, char_to_idx, idx_to_char, transform, max_len=120):
    model.eval()
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        output = output.softmax(2).argmax(2).squeeze(0)
        pred = []
        for idx in output:
            char = idx_to_char.get(idx.item(), '')
            if char and char not in ['<S>', '<E>', '<PAD>'] and (not pred or char != pred[-1]):
                pred.append(char)
            if char == '<E>':
                break
        return ''.join(pred)

# --- Test Inference ---
test_image_path = os.path.join(images_dir, image_formula_pairs[0][0])
predicted_latex = predict_latex(test_image_path, model, char_to_idx, idx_to_char, transform)
print(f"Predicted LaTeX: {predicted_latex}")
print(f"Ground Truth: {image_formula_pairs[0][1]}")
