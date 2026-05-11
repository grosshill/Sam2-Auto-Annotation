from argparse import ArgumentParser
from ultralytics import YOLO


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Inspect classes in a YOLO weight file.")
    parser.add_argument("--weights", default="./yolo26n.pt", help="Path to YOLO .pt weight file")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["drone", "uav", "quadcopter", "airplane", "helicopter"],
        help="Class name keywords to search for",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    print("All classes:")
    for idx, name in names.items():
        print(f"{idx:>2}: {name}")

    print(f"\nTotal classes: {len(names)}")
    matches = [(i, n) for i, n in names.items() if any(k.lower() in n.lower() for k in args.keywords)]
    print(f"Keyword matches ({', '.join(args.keywords)}): {matches if matches else 'None'}")


if __name__ == "__main__":
    main()
