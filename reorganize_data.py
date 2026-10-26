import os
import re
import shutil
from pathlib import Path


def reorganize_trackman_data(data_dir="data", dry_run=False):
    """
    'data/YYYY-MM-DD range/data.csv' を探索し、
    'data/YYYY-MM-DD.csv' としてコピー（抽出）する。
    """
    root = Path(data_dir)
    if not root.exists():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    # 'YYYY-MM-DD range' パターンの正規表現
    folder_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+range$")

    migrated_count = 0
    skipped_count = 0

    # data 直下のディレクトリを走査
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    print(f"--- Reorganizing data in '{data_dir}' (dry_run={dry_run}) ---")

    for folder in subdirs:
        match = folder_pattern.match(folder.name)
        if not match:
            continue

        date_str = match.group(1)
        target_csv_name = f"{date_str}.csv"
        target_path = root / target_csv_name

        # フォルダ内の CSV を探索（data.csv を優先、なければ最初の CSV）
        csv_candidates = list(folder.glob("*.csv"))
        source_csv = None

        data_csv = folder / "data.csv"
        if data_csv.exists():
            source_csv = data_csv
        elif csv_candidates:
            source_csv = sorted(csv_candidates)[0]

        if not source_csv:
            print(f"[Warning] No CSV found in {folder.name}, skipping.")
            skipped_count += 1
            continue

        if dry_run:
            print(f"[Plan] {source_csv} -> {target_path}")
        else:
            # 既に存在する場合は上書きコピー
            shutil.copy2(source_csv, target_path)
            print(f"[Copied] {source_csv.name} -> {target_path.name}")

        migrated_count += 1

    print("--------------------------------------------------")
    print(f"Total processed: {migrated_count} files, Skipped: {skipped_count}")

    if not dry_run and migrated_count > 0:
        print("\nCSV files have been extracted successfully.")
        print("To remove old 'YYYY-MM-DD range' folders and legacy files, run:")
        print("  python reorganize_data.py --cleanup")


def cleanup_legacy_folders(data_dir="data"):
    """
    'data/YYYY-MM-DD range' フォルダを削除する。
    対応する 'YYYY-MM-DD.csv' が存在することを確認した安全な削除。
    """
    root = Path(data_dir)
    folder_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+range$")

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    deleted_count = 0

    print(f"--- Cleaning up old folders in '{data_dir}' ---")
    for folder in subdirs:
        match = folder_pattern.match(folder.name)
        if not match:
            continue

        date_str = match.group(1)
        expected_csv = root / f"{date_str}.csv"

        # 対応する新形式 CSV が存在する場合のみフォルダごと安全に削除
        if expected_csv.exists():
            shutil.rmtree(folder)
            print(f"[Removed folder] {folder.name}")
            deleted_count += 1
        else:
            print(
                f"[Kept] {folder.name} (Missing corresponding {expected_csv.name})"
            )

    print("--------------------------------------------------")
    print(f"Cleanup finished. Removed {deleted_count} folder(s).")


if __name__ == "__main__":
    import sys

    if "--cleanup" in sys.argv:
        confirm = input(
            "Are you sure you want to delete all 'YYYY-MM-DD range' folders? (y/N): "
        )
        if confirm.strip().lower() == "y":
            cleanup_legacy_folders("data")
        else:
            print("Cleanup cancelled.")
    elif "--dry-run" in sys.argv:
        reorganize_trackman_data("data", dry_run=True)
    else:
        reorganize_trackman_data("data", dry_run=False)