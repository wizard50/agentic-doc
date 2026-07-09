#!/usr/bin/env python
import subprocess
from pathlib import Path


def download_rust_book(target_dir: str = "data/download/rust-book"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/rust-lang/book.git"

    if (target_path / ".git").exists():
        print("Rust book already cloned. Pulling latest...")
        subprocess.run(["git", "-C", str(target_path), "pull"], check=True)
    else:
        print(f"Cloning Rust book to {target_path}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_path)], check=True)

    src_dir = target_path / "src"
    if src_dir.exists():
        print(f"✅ Rust Book src/ ready at: {src_dir}")
        print(f"   Files: {len(list(src_dir.glob('**/*.md')))} markdown files")
    else:
        print("⚠️ src/ folder not found!")


if __name__ == "__main__":
    download_rust_book()
