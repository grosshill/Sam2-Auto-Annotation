from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Quick YOLO dataset sanity checks.")
    parser.add_argument("--root", default="./datasets/drone", help="Dataset root directory")
    return parser


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.glob("**/*") if p.suffix.lower() in exts])


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)

    train_images = list_images(root / "images" / "train")
    val_images = list_images(root / "images" / "val")
    total = len(train_images) + len(val_images)

    print(f"Dataset root: {root.resolve()}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    missing_labels = []
    for img in train_images + val_images:
        split = "train" if "images/train" in img.as_posix() else "val"
        label = root / "labels" / split / (img.stem + ".txt")
        if not label.exists():
            missing_labels.append((img, label))

    print(f"Total images: {total}")
    print(f"Missing labels: {len(missing_labels)}")
    if missing_labels:
        print("First 10 missing label pairs:")
        for img, label in missing_labels[:10]:
            print(f"  image={img} label={label}")


if __name__ == "__main__":
    main()

