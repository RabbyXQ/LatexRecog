import torch
from PIL import Image
from torchvision import transforms
import json
import os
from model import CRNN  # Assuming you saved your model class in model.py
import kagglehub

# --- Download and load dataset ---
path = kagglehub.dataset_download("gregoryeritsyan/im2latex-230k")
dataset_path = os.path.join(path, "PRINTED_TEX_230k")

# --- JSON file contains vocab mapping ---
json_file = os.path.join(dataset_path, "230k.json")

# --- Load vocab ---
with open(json_file, 'r') as f:
    vocab = json.load(f)
vocab = {k: int(v) for k, v in vocab.items()}
char_to_idx = vocab
idx_to_char = {idx: char for char, idx in vocab.items()}

# --- Adjust vocab_size to match checkpoint (581 tokens) ---
vocab_size = len(vocab)  # This should be 581 if the checkpoint was trained with 581 tokens

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(vocab_size).to(device)

# --- Load model state dict with vocab size mismatch handling ---
state_dict = torch.load('latex_recog_model.pth', map_location=device)

# Remove the `fc` layer weights and bias from the state dict to handle size mismatch
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)

# Load the rest of the state dict
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor()
])

# --- Prediction Function ---
def predict_latex(image_path):
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0).to(device)
    
    # Debugging: Check image shape
    print(f"Image shape: {img.shape}")

    with torch.no_grad():
        output = model(img)

        # Debugging: Check output shape before softmax
        print(f"Output shape (before softmax): {output.shape}")

        output = output.softmax(2).argmax(2).squeeze(0)

        # Debugging: Check output after softmax
        print(f"Output after softmax: {output}")

        pred = []
        for idx in output:
            char = idx_to_char.get(idx.item(), '')
            if char and char not in ['<S>', '<E>', '<PAD>'] and (not pred or char != pred[-1]):
                pred.append(char)
            if char == '<E>':
                break
        
        # Debugging: Check the final prediction
        print(f"Final Prediction: {''.join(pred)}")
        return ''.join(pred)

# --- App Flow ---
if __name__ == "__main__":
    image_path = "./c0200564-800px-wm.jpg"
    if not os.path.exists(image_path):
        print("Image not found. Please check the path.")
    else:
        latex = predict_latex(image_path)
        print("\nPredicted LaTeX Code:")
        print(latex)
