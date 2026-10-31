"""
01_prepare_data.py  -  MIT-BIH  ->  64x64 grayscale beat images (paper Section 3.1)

    raw ECG (lead II) -> band-pass 0.1-100 Hz -> min-max to [0,1] -> 256-sample slice around every
    annotated R peak -> AAMI label -> draw the slice as a line plot -> 64x64 uint8 image

Output: outputs/ecg_images.npz  (X uint8 [N,64,64], y int8 [N], rec, sample)
"""
import numpy as np
import cv2
import wfdb
from scipy.signal import butter, sosfiltfilt
import config as C


def find_or_get_database():
    """Sets C.MITDB_DIR to a folder that holds 100.hea ... 234.hea.
    Order: 1) folder already there (also searched inside data/)  2) a .zip inside data/ (e.g. the one from PhysioNet)
           3) download with wfdb."""
    import zipfile
    def look():
        for base in (C.MITDB_DIR, C.DATA_DIR):
            if base.exists():
                for h in base.rglob("100.hea"):
                    return h.parent
        return None
    d = look()
    if d is None:
        for z in C.DATA_DIR.glob("*.zip"):
            print(f"Extracting {z.name} ...")
            zipfile.ZipFile(z).extractall(C.DATA_DIR)
            d = look()
            if d: break
    if d is None:
        print("Downloading MIT-BIH Arrhythmia Database from PhysioNet (about 100 MB) ...")
        wfdb.dl_database("mitdb", str(C.MITDB_DIR))
        d = look()
    C.MITDB_DIR = d


def load_record(name):
    """Returns (signal, R-peak sample indices, beat symbols) or None if lead II is missing."""
    path = str(C.MITDB_DIR / name)
    rec = wfdb.rdrecord(path)
    if C.LEAD in rec.sig_name:
        idx = rec.sig_name.index(C.LEAD)
    elif C.LEAD_FALLBACK:                   # records 102 and 104 only carry V5 / V2
        idx = 0
        print(f"  {name}: no {C.LEAD} ({rec.sig_name}) -> using channel 0 ({rec.sig_name[0]}) instead")
    else:
        print(f"  {name}: no {C.LEAD} lead ({rec.sig_name}) -> skipped")
        return None
    sig = rec.p_signal[:, idx].astype(np.float64)
    ann = wfdb.rdann(path, "atr")
    return sig, ann.sample, ann.symbol


def preprocess(sig):
    """Zero-phase band-pass 0.1-100 Hz, then min-max normalise the recording to [0, 1]."""
    sos = butter(3, C.BANDPASS, btype="bandpass", fs=C.FS, output="sos")
    f = sosfiltfilt(sos, sig)
    if C.NORM_MODE == "record":
        f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    return f.astype(np.float32)


def extract_beats(sig, samples, symbols):
    """256-sample windows centred on the annotated R peaks that belong to an AAMI class."""
    half = C.BEAT_LEN // 2
    beats, labels, locs = [], [], []
    for s, sym in zip(samples, symbols):
        if sym not in C.AAMI_MAP:            # '+', '~', '|', 'x' ... are not beats
            continue
        a, b = s - half, s + half
        if a < 0 or b > len(sig):            # window would run off the recording
            continue
        w = sig[a:b]
        if C.NORM_MODE == "beat":
            w = (w - w.min()) / (w.max() - w.min() + 1e-12)
        beats.append(w)
        labels.append(C.AAMI_MAP[sym])
        locs.append(s)
    return beats, labels, locs


def beat_to_image(beat):
    """Draw the beat as a polyline (time on x, amplitude on y) and shrink it to 64x64."""
    S = C.RENDER_SIZE
    x = np.linspace(0, S - 1, len(beat))
    y = (1.0 - np.clip(beat, 0, 1)) * (S - 1)                      # big amplitude = top of the image
    pts = np.stack([x, y], 1).round().astype(np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((S, S), np.uint8)
    cv2.polylines(canvas, [pts], False, 255, thickness=C.LINE_THICKNESS, lineType=cv2.LINE_AA)
    img = cv2.resize(canvas, (C.IMG, C.IMG), interpolation=cv2.INTER_AREA)
    return 255 - img if C.DARK_ON_LIGHT else img


def main():
    find_or_get_database()
    names = sorted(p.stem for p in C.MITDB_DIR.glob("*.hea"))
    print(f"{len(names)} records found in {C.MITDB_DIR}")

    all_beats, all_y, all_rec, all_loc = [], [], [], []
    for name in names:
        r = load_record(name)
        if r is None:
            continue
        sig, samples, symbols = r
        b, l, loc = extract_beats(preprocess(sig), samples, symbols)
        all_beats += b; all_y += l; all_loc += loc; all_rec += [name] * len(l)
        print(f"  {name}: {len(l):5d} beats")

    y = np.array(all_y, dtype=np.int8)
    rec = np.array(all_rec)
    loc = np.array(all_loc)

    # keep only N_CAP of the (huge) N class, like the paper (34,000)
    keep = np.arange(len(y))
    if C.N_CAP is not None:
        rng = np.random.default_rng(C.SEED)
        n_idx = np.where(y == 0)[0]
        if len(n_idx) > C.N_CAP:
            drop = set(rng.choice(n_idx, len(n_idx) - C.N_CAP, replace=False).tolist())
            keep = np.array([i for i in keep if i not in drop])
    y, rec, loc = y[keep], rec[keep], loc[keep]
    beats = [all_beats[i] for i in keep]

    print("\nRendering images ...")
    X = np.stack([beat_to_image(b) for b in beats]).astype(np.uint8)

    np.savez_compressed(C.OUT_DIR / "ecg_images.npz", X=X, y=y, rec=rec, sample=loc)
    counts = np.bincount(y, minlength=5)
    print("\nClass counts:", dict(zip(C.CLASSES, counts.tolist())), " total =", len(y))
    print("Saved", C.OUT_DIR / "ecg_images.npz", " X:", X.shape, X.dtype)

    # quick visual check: 5 examples of every class
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 6, figsize=(9, 8))
    for c in range(5):
        ids = np.where(y == c)[0][:6]
        for j in range(6):
            ax[c, j].axis("off")
            if j < len(ids):
                ax[c, j].imshow(X[ids[j]], cmap="gray", vmin=0, vmax=255)
        ax[c, 0].set_title(C.CLASSES[c], loc="left")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "sample_beats.png", dpi=120)


if __name__ == "__main__":
    main()
