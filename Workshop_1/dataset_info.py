"""
dataset_info.py
---------------
Script to scan and validate dataset integrity.
Verify checksum (MD5 y SHA-256), filesize, binary header IDX y classes distributions.
"""

import gzip
import hashlib
from pathlib import Path
import struct
import numpy as np

# Official control checksums published by Zalando Research
OFFICIAL_CHECKSUMS = {
    "train-images-idx3-ubyte.gz": {
        "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "sha256": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    },
    "train-labels-idx1-ubyte.gz": {
        "md5": "25c81989df183df01b3e8a0aad5dffbe",
        "sha256": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    },
    "t10k-images-idx3-ubyte.gz": {
        "md5": "bef4ecab320f06d8554ea6380940ec79",
        "sha256": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    },
    "t10k-labels-idx1-ubyte.gz": {
        "md5": "bb300cfdad3c16e7a12a480ee83cd310",
        "sha256": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    },
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def compute_file_hashes(file_path: Path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def read_idx_images(file_path: Path):
    with gzip.open(file_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_images = struct.unpack(">I", f.read(4))[0]
        n_rows = struct.unpack(">I", f.read(4))[0]
        n_cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, n_rows, n_cols)
    return magic, n_images, (n_rows, n_cols), data


def read_idx_labels(file_path: Path):
    with gzip.open(file_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_labels = struct.unpack(">I", f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return magic, n_labels, labels


def inspect_dataset(data_dir: Path):
    print("=" * 70)
    print("Resume and integrity of dataset")
    print(f"Directory analized: {data_dir.resolve()}")
    print("=" * 70)

    if not data_dir.exists():
        print(f"Error: The directory {data_dir} does not exist.")
        return

    gz_files = sorted(list(data_dir.glob("*.gz")))
    if not gz_files:
        print(f"No files .gz where found on {data_dir}.")
        return

    total_bytes = sum(f.stat().st_size for f in gz_files)
    print(f"Files found: {len(gz_files)}")
    print(f"Total size on disk: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes:,} bytes)")
    print("-" * 70)

    # Check sums validations
    print("Checksum verification (HASH CHECKSUMS):")
    all_ok = True
    for file_path in gz_files:
        name = file_path.name
        md5_calc, sha256_calc = compute_file_hashes(file_path)
        expected = OFFICIAL_CHECKSUMS.get(name)

        status_md5 = "CORRECT" if expected and md5_calc == expected["md5"] else "UNKNOWN/DIFFERENT"
        status_sha = "CORRECT" if expected and sha256_calc == expected["sha256"] else "UNKNOWN/DIFFERENT"

        print(f"\n* File: {name}")
        print(f"  Size: {file_path.stat().st_size:,} bytes")
        print(f"  MD5:    {md5_calc} [{status_md5}]")
        print(f"  SHA256: {sha256_calc} [{status_sha}]")

        if status_md5 != "CORRECT" or status_sha != "CORRECT":
            all_ok = False

    print("\n" + "-" * 70)
    print(f"Global state integrity: {'FULL AND VERIFIED' if all_ok else 'REQUIRES REVISION'}")
    print("-" * 70)

    # Load data for structural statistics
    train_img_path = data_dir / "train-images-idx3-ubyte.gz"
    train_lbl_path = data_dir / "train-labels-idx1-ubyte.gz"
    test_img_path = data_dir / "t10k-images-idx3-ubyte.gz"
    test_lbl_path = data_dir / "t10k-labels-idx1-ubyte.gz"

    if all(p.exists() for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]):
        m_t_img, n_t_img, shape_t, X_train = read_idx_images(train_img_path)
        m_t_lbl, n_t_lbl, y_train = read_idx_labels(train_lbl_path)
        m_k_img, n_k_img, shape_k, X_test = read_idx_images(test_img_path)
        m_k_lbl, n_k_lbl, y_test = read_idx_labels(test_lbl_path)

        print("\n Structural Metadata:")
        print(f"  Training: {n_t_img:,} images of {shape_t[0]}x{shape_t[1]} pixels (Magic: {m_t_img})")
        print(f"  Labels: {n_t_lbl:,} registries (Magic: {m_t_lbl})")
        print(f"  Test:        {n_k_img:,} images of {shape_k[0]}x{shape_k[1]} pixels (Magic: {m_k_img})")
        print(f"  Labels: {n_k_lbl:,} registries (Magic: {m_k_lbl})")
        print(f"  Pixels Range: min={X_train.min()}, max={X_train.max()} (type {X_train.dtype})")
        print(f"  Labels Range: min={y_train.min()}, max={y_train.max()} (type {y_train.dtype})")

        print("\nDISTRIBUTION OF SAMPLES PER CLASS:")
        print(f"{'Label':<10}{'Class Name':<20}{'Training':<16}{'Test':<10}")
        print("-" * 56)
        train_counts = np.bincount(y_train, minlength=10)
        test_counts = np.bincount(y_test, minlength=10)
        for i in range(10):
            print(f"{i:<10}{CLASS_NAMES[i]:<20}{train_counts[i]:<16}{test_counts[i]:<10}")
        print("-" * 56)
        print(f"{'Total':<30}{sum(train_counts):<16}{sum(test_counts):<10}")

    print("=" * 70)


if __name__ == "__main__":
    # Search data en data/raw or in code/data
    base_dir = Path(__file__).resolve().parent
    primary_dir = base_dir / "data" / "raw"
    fallback_dir = base_dir / "code" / "data"
    target_dir = primary_dir if primary_dir.exists() and any(primary_dir.glob("*.gz")) else fallback_dir

    inspect_dataset(target_dir)
