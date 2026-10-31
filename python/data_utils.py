"""data_utils.py - load the image dataset and produce the SAME split in every script."""
import numpy as np
from sklearn.model_selection import train_test_split
import config as C


def load_xy():
    """Returns Xq (N,64,64) uint8 with values 0..15 (UINT4 pixels) and y (N,) int64."""
    d = np.load(C.OUT_DIR / "ecg_images.npz")
    x8, y = d["X"], d["y"].astype(np.int64)
    xq = np.round(x8.astype(np.float32) / 255.0 * 15.0).astype(np.uint8)   # 0..255 -> 0..15
    return xq, y


def split_indices(y):
    """Deterministic stratified split: 85 % train / 15 % test, then 15 % of train -> validation."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    tr, va = train_test_split(tr, test_size=C.VAL_SPLIT, stratify=y[tr], random_state=C.SEED)
    return tr, va, te


def to_float(xq):
    """UINT4 pixels -> float network input. x/15 means the float model sees exactly what the FPGA sees."""
    return (xq.astype(np.float32) / 15.0)[..., None]
