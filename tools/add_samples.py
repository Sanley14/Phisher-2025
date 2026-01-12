
import argparse
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
import sys

REQUIRED_COLS = ["email_text", "language", "label"]


def backup_file(path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".{ts}.bak")
    shutil.copy2(path, dest)
    return dest


def validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def append_csv(dataset_path: Path, new_csv: Path, dry_run: bool = True, backup: bool = True):
    ds = pd.read_csv(dataset_path)
    validate_df(ds)

    new = pd.read_csv(new_csv)
    validate_df(new)

    print(f"Dataset rows: {len(ds)}, New rows: {len(new)}")
    if dry_run:
        print("Dry-run: no changes written. Sample of new rows:")
        print(new.head(10).to_string(index=False))
        return 0

    if backup:
        bak = backup_file(dataset_path)
        print(f"Backup created at: {bak}")

    combined = pd.concat([ds, new], ignore_index=True)
    combined.to_csv(dataset_path, index=False)
    print(f"Appended {len(new)} rows. New dataset size: {len(combined)}")
    return 0


def add_single(dataset_path: Path, text: str, language: str, label: int, dry_run: bool = True, backup: bool = True):
    ds = pd.read_csv(dataset_path)
    validate_df(ds)

    new = pd.DataFrame([{"email_text": text, "language": language, "label": int(label)}])
    print("New row:")
    print(new.to_string(index=False))
    if dry_run:
        print("Dry-run: no changes written.")
        return 0

    if backup:
        bak = backup_file(dataset_path)
        print(f"Backup created at: {bak}")

    combined = pd.concat([ds, new], ignore_index=True)
    combined.to_csv(dataset_path, index=False)
    print(f"Appended 1 row. New dataset size: {len(combined)}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to target dataset CSV (will be updated)')
    parser.add_argument('--add', help='Path to CSV file containing rows to append')
    parser.add_argument('--text', help='Single email text to append')
    parser.add_argument('--language', help='Language code for single sample (en, zh, ru, sw)', default='en')
    parser.add_argument('--label', type=int, choices=[0,1], help='Label for single sample (0=legit, 1=phish)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    parser.add_argument('--backup', action='store_true', help='Create a timestamped backup before modifying dataset')

    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        sys.exit(2)

    try:
        if args.add:
            rc = append_csv(dataset, Path(args.add), dry_run=args.dry_run, backup=args.backup)
            sys.exit(rc)
        elif args.text and (args.label is not None):
            rc = add_single(dataset, args.text, args.language, args.label, dry_run=args.dry_run, backup=args.backup)
            sys.exit(rc)
        else:
            print("Either --add <csv> or --text and --label must be provided.")
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(3)
