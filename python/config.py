"""
config.py - every knob of the pipeline lives here.
Numbers marked (paper) come from Mangaraj, Mahapatra & Ari, BSPC 109 (2025) 108063.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MITDB_DIR = Path(os.environ.get("MITDB_DIR", DATA_DIR / "mitdb"))   # where the .dat/.hea/.atr files live
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "outputs"))         # everything we produce
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data / pre-processing
FS = 360                      # MIT-BIH sampling rate in Hz (paper)
BANDPASS = (0.1, 100.0)       # band-pass edges in Hz (paper)
BEAT_LEN = 256                # samples per beat slice, centred on the R peak (paper)
IMG = 64                      # final image is IMG x IMG grayscale (paper)
RENDER_SIZE = 256             # beat is first drawn on a 256x256 canvas, then shrunk to IMG
LINE_THICKNESS = 6            # pen width on the 256x256 canvas (about 1.5 px after shrinking)
DARK_ON_LIGHT = True          # True: black waveform on white paper (like a matplotlib plot)
NORM_MODE = "record"          # "record": min-max over the whole recording (paper) | "beat": per beat
N_CAP = 34000                 # paper keeps 34,000 of the N-class beats; set None to keep all
LEAD = "MLII"                 # lead II (paper)
LEAD_FALLBACK = True          # records 102 & 104 have no MLII: True = use their first channel (V5) so the
                              # paced-beat count is close to the paper's (Q = 7772); False = skip them

# AAMI EC57 grouping (paper, Table 2): symbol -> class id
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {}
for s in "NLRej": AAMI_MAP[s] = 0      # N: normal, LBBB, RBBB, atrial escape, nodal escape
for s in "AaJS":  AAMI_MAP[s] = 1      # S: supraventricular ectopic
for s in "VE":    AAMI_MAP[s] = 2      # V: ventricular ectopic
for s in "F":     AAMI_MAP[s] = 3      # F: fusion
for s in "/fQ":   AAMI_MAP[s] = 4      # Q: paced, fusion of paced, unclassifiable

# ---------------------------------------------------------------- training
SEED = 42
TEST_SIZE = 0.15              # 85:15 train:test (paper)
VAL_SPLIT = 0.15              # 15 % of the training part is used for validation (paper)
LR = 1e-4                     # Adam learning rate (paper)
BATCH = 64                    # (paper)
EPOCHS = int(os.environ.get("EPOCHS", 100))   # (paper)
L2 = 1e-4                     # paper says L2 was used but not the value - this is my choice
USE_CLASS_WEIGHTS = False     # paper did not rebalance; flip to True if S/F recall is poor
