"""
03_test.py  -  evaluate the trained float model on the held-out 15 % test set (paper Tables 4-5, Fig. 13)
"""
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data_utils import load_xy, split_indices, to_float

xq, y = load_xy()
_, _, te = split_indices(y)
x_te, y_te = to_float(xq[te]), y[te]

model = keras.models.load_model(C.OUT_DIR / "best_model.keras", compile=False)
prob = model.predict(x_te, batch_size=256, verbose=0)
pred = prob.argmax(1)

acc = accuracy_score(y_te, pred)
print(f"\nTest accuracy (FP32): {acc * 100:.2f} %   on {len(te)} beats   (paper: 97.79 %)\n")
report = classification_report(y_te, pred, target_names=C.CLASSES, digits=4, zero_division=0)
print(report)
cm = confusion_matrix(y_te, pred)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)

open(C.OUT_DIR / "test_report.txt", "w").write(f"accuracy {acc:.4f}\n\n{report}\n{cm}\n")
np.save(C.OUT_DIR / "test_probs.npy", prob)

# ---- confusion-matrix figure (counts and row-normalised)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, mat, title, fmt in ((ax[0], cm, "counts", "d"),
                           (ax[1], cm / cm.sum(1, keepdims=True), "row-normalised", ".2f")):
    im = a.imshow(mat, cmap="Blues")
    a.set_xticks(range(5)); a.set_xticklabels(C.CLASSES)
    a.set_yticks(range(5)); a.set_yticklabels(C.CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"Confusion matrix - {title}")
    for i in range(5):
        for j in range(5):
            a.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                   color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
plt.tight_layout(); plt.savefig(C.OUT_DIR / "confusion_matrix.png", dpi=130)
