"""
dataset_info.py
---------------
Script para escanear y validar la integridad del conjunto de datos Fashion-MNIST.
Verifica sumas de comprobación (MD5 y SHA-256), tamaños de archivo, cabeceras
binarias IDX y distribución de clases.
"""

import gzip
import hashlib
from pathlib import Path
import struct
import numpy as np

# Sumas oficiales de control publicadas por Zalando Research
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
    print("RESUMEN E INTEGRIDAD DEL CONJUNTO DE DATOS: FASHION-MNIST")
    print(f"Directorio analizado: {data_dir.resolve()}")
    print("=" * 70)

    if not data_dir.exists():
        print(f"Error: El directorio {data_dir} no existe.")
        return

    gz_files = sorted(list(data_dir.glob("*.gz")))
    if not gz_files:
        print(f"No se encontraron archivos .gz en {data_dir}.")
        return

    total_bytes = sum(f.stat().st_size for f in gz_files)
    print(f"Archivos encontrados: {len(gz_files)}")
    print(f"Tamano total en disco: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes:,} bytes)")
    print("-" * 70)

    # Verificacion de sumas de comprobacion
    print("VERIFICACION DE SUMAS DE CONTROL (HASH CHECKSUMS):")
    all_ok = True
    for file_path in gz_files:
        name = file_path.name
        md5_calc, sha256_calc = compute_file_hashes(file_path)
        expected = OFFICIAL_CHECKSUMS.get(name)

        status_md5 = "CORRECTO" if expected and md5_calc == expected["md5"] else "DESCONOCIDO/DISCREPANTE"
        status_sha = "CORRECTO" if expected and sha256_calc == expected["sha256"] else "DESCONOCIDO/DISCREPANTE"

        print(f"\n* Archivo: {name}")
        print(f"  Tamano: {file_path.stat().st_size:,} bytes")
        print(f"  MD5:    {md5_calc} [{status_md5}]")
        print(f"  SHA256: {sha256_calc} [{status_sha}]")

        if status_md5 != "CORRECTO" or status_sha != "CORRECTO":
            all_ok = False

    print("\n" + "-" * 70)
    print(f"Estado global de integridad: {'INTEGRO Y VERIFICADO' if all_ok else 'REQUIERE REVISION'}")
    print("-" * 70)

    # Carga de datos para estadísticas estructurales
    train_img_path = data_dir / "train-images-idx3-ubyte.gz"
    train_lbl_path = data_dir / "train-labels-idx1-ubyte.gz"
    test_img_path = data_dir / "t10k-images-idx3-ubyte.gz"
    test_lbl_path = data_dir / "t10k-labels-idx1-ubyte.gz"

    if all(p.exists() for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]):
        m_t_img, n_t_img, shape_t, X_train = read_idx_images(train_img_path)
        m_t_lbl, n_t_lbl, y_train = read_idx_labels(train_lbl_path)
        m_k_img, n_k_img, shape_k, X_test = read_idx_images(test_img_path)
        m_k_lbl, n_k_lbl, y_test = read_idx_labels(test_lbl_path)

        print("\nMETADATOS ESTRUCTURALES:")
        print(f"  Entrenamiento: {n_t_img:,} imagenes de {shape_t[0]}x{shape_t[1]} pixeles (Magic: {m_t_img})")
        print(f"  Etiquetas ent: {n_t_lbl:,} registros (Magic: {m_t_lbl})")
        print(f"  Prueba:        {n_k_img:,} imagenes de {shape_k[0]}x{shape_k[1]} pixeles (Magic: {m_k_img})")
        print(f"  Etiquetas pba: {n_k_lbl:,} registros (Magic: {m_k_lbl})")
        print(f"  Rango de pixeles: min={X_train.min()}, max={X_train.max()} (tipo {X_train.dtype})")
        print(f"  Rango de etiquetas: min={y_train.min()}, max={y_train.max()} (tipo {y_train.dtype})")

        print("\nDISTRIBUCION DE MUESTRAS POR CLASE:")
        print(f"{'Etiqueta':<10}{'Nombre de Clase':<20}{'Entrenamiento':<16}{'Prueba':<10}")
        print("-" * 56)
        train_counts = np.bincount(y_train, minlength=10)
        test_counts = np.bincount(y_test, minlength=10)
        for i in range(10):
            print(f"{i:<10}{CLASS_NAMES[i]:<20}{train_counts[i]:<16}{test_counts[i]:<10}")
        print("-" * 56)
        print(f"{'Total':<30}{sum(train_counts):<16}{sum(test_counts):<10}")

    print("=" * 70)


if __name__ == "__main__":
    # Buscar datos en data/raw o en code/data
    base_dir = Path(__file__).resolve().parent
    primary_dir = base_dir / "data" / "raw"
    fallback_dir = base_dir / "code" / "data"
    target_dir = primary_dir if primary_dir.exists() and any(primary_dir.glob("*.gz")) else fallback_dir

    inspect_dataset(target_dir)
