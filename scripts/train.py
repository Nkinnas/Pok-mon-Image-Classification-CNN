
import json, time
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", message="Palette images with Transparency")

from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   #tolerates slightly broken images

class ToRGB:
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")



def get_loaders(data_dir="data", img_size=224, batch_size=256, workers=8):
    train_tfms = transforms.Compose([
        ToRGB(),                          
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        ToRGB(),                            
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir)/"train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(Path(data_dir)/"val",   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(Path(data_dir)/"test",  transform=eval_tfms)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True, persistent_workers=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True),
    }
    idx_to_class = {v:k for k,v in train_ds.class_to_idx.items()}
    return loaders, idx_to_class


def build_model(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m

def evaluate(model, loader, device, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast("cuda"):
                out = model(x)
            loss_sum += criterion(out,y).item() * x.size(0)
            correct += (out.argmax(1)==y).sum().item()
            tot += y.size(0)
    return loss_sum/tot, correct/tot

def main(data_dir="data", epochs=30, batch_size=256, lr=3e-3, wd=1e-4, img_size=224, save_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loaders, idx_to_class = get_loaders(data_dir, img_size, batch_size)
    num_classes = len(idx_to_class)
    print("Classes:", num_classes)

    torch.backends.cudnn.benchmark = True

    model = build_model(num_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = OneCycleLR(opt, max_lr=lr, epochs=epochs,
                           steps_per_epoch=len(loaders["train"]))
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(save_dir)/"best.pt"
    with open(Path(save_dir)/"idx_to_class.json","w") as f: json.dump(idx_to_class,f,indent=2)

    best_acc = 0.0
    for ep in range(1, epochs+1):
        t0=time.time()
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in tqdm(loaders["train"], desc=f"Epoch {ep:02d}", ncols=100):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)

        tr_loss, tr_acc = loss_sum/tot, correct/tot
        val_loss, val_acc = evaluate(model, loaders["val"], device, crit)
        print(f"[epoch {ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}  time {time.time()-t0:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "model_name":"resnet18",
                        "idx_to_class": idx_to_class}, best_path)
            print("Saved best ->", best_path)

    #test best
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, loaders["test"], device, crit)
    print(f"TEST acc={test_acc:.4f} loss={test_loss:.4f}")

if __name__ == "__main__":
    main()
