
import torch
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_DIR = Path(__file__).resolve().parent.parent / "Image_to_predict"

#Paths to checkpoint and class index map
CKPT_PATH = Path("checkpoints/best.pt")
IDX_PATH = Path("checkpoints/idx_to_class.json")

#Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load idx_to_class mapping
with open(IDX_PATH, "r") as f:
    idx_to_class = json.load(f)
num_classes = len(idx_to_class)

# Build model skeleton
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)

#Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

#Handle string/int keys in idx_to_class
def get_class_name(idx):
    return idx_to_class.get(str(idx)) or idx_to_class.get(idx)

# Find all images in the folder
images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]

if not images:
    print(f"No images found in {IMAGE_DIR}")
else:
    print(f"\nFound {len(images)} image(s) in {IMAGE_DIR}\n")
    for img_path in sorted(images):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.max(probs, dim=0)

        pred_class = get_class_name(top_idx.item())
        confidence = top_prob.item() * 100
        print(f"{img_path.name}: {pred_class} ({confidence:.2f}% confidence)")
