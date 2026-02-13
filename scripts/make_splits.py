
import argparse, random, shutil
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def is_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im: im.verify()
        return True
    except Exception:
        return False

def copy_split(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="PokemonCNN/Pokemon_images",
                    help="Folder with 151 class subfolders")
    ap.add_argument("--dst", default="PokemonCNN/data",
                    help="Output root containing train/val/test")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1.0"
    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    for s in ("train","val","test"):
        (dst / s).mkdir(parents=True, exist_ok=True)

    classes = sorted([p for p in src.iterdir() if p.is_dir()])
    print(f"Found {len(classes)} classes in {src.resolve()}")

    for cls in classes:
        imgs = [p for p in cls.rglob("*") if p.suffix.lower() in IMG_EXTS]
        imgs = [p for p in imgs if is_ok(p)]
        if not imgs:
            print(f"[WARN] {cls.name} has no valid images, skipping."); continue

        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * args.train)
        n_val = int(n * args.val)
        n_te = n - n_tr - n_val

        splits = {
            "train": imgs[:n_tr],
            "val":   imgs[n_tr:n_tr+n_val],
            "test":  imgs[n_tr+n_val:]
        }
        for split, files in splits.items():
            copy_split(files, dst / split / cls.name)

        print(f"{cls.name:18s} total={n:4d}  train={n_tr:4d}  val={n_val:4d}  test={n_te:4d}")

    print(f"\nDone. Data written under {dst.resolve()}")

if __name__ == "__main__":
    main()
