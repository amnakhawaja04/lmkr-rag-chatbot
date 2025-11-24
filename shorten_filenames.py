import os
import hashlib
from pathlib import Path

# Folder where your LMKR .txt files are
DATA_DIR = Path("C:/Users/afarooq/Downloads/lmkr-rag-chatbot/data/lmkr")


# Maximum file name length (without path) we allow
MAX_NAME_LENGTH = 100  # you can adjust if you want


def shorten_filename(name: str, max_length: int = MAX_NAME_LENGTH) -> str:
    """
    Shortens a filename (without path) to max_length characters.
    Keeps the extension and adds a short hash for uniqueness.
    """
    # Split extension
    if "." in name:
        base, ext = name.rsplit(".", 1)
        ext = "." + ext
    else:
        base, ext = name, ""

    # If already short enough, return as is
    if len(name) <= max_length:
        return name

    # Create a small hash from the original name
    hash_suffix = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]

    # How many chars can we use for the base part?
    # Reserve space for '_' + hash + extension
    reserved = 1 + len(hash_suffix) + len(ext)
    max_base_len = max_length - reserved
    if max_base_len <= 0:
        # Fallback: just use the hash and extension
        return f"{hash_suffix}{ext}"

    short_base = base[:max_base_len]
    new_name = f"{short_base}_{hash_suffix}{ext}"
    return new_name


def main():
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR.resolve()}")
        return

    for path in DATA_DIR.glob("*"):
        if not path.is_file():
            continue

        old_name = path.name
        new_name = shorten_filename(old_name, MAX_NAME_LENGTH)

        if old_name == new_name:
            # No need to rename
            continue

        new_path = path.with_name(new_name)

        # Avoid overwriting existing files
        if new_path.exists():
            print(f"Skipping (target exists): {old_name} -> {new_name}")
            continue

        print(f"Renaming: {old_name}")
        print(f"      -> {new_name}")
        path.rename(new_path)

    print("Done renaming long filenames.")


if __name__ == "__main__":
    main()
