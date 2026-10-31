"""
02_train.py  -  train the dual-branch CNN in float32 (paper Section 4.1/4.2)

Adam, lr 1e-4, batch 64, 100 epochs, sparse categorical cross-entropy, L2 regularisation.
The checkpoint with the best validation accuracy is kept as outputs/best_model.keras
"""
import json
import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data_utils import load_xy, split_indices, to_float
from model_def import build_model

keras.utils.set_random_seed(C.SEED)

xq, y = load_xy()
tr, va, te = split_indices(y)
x_tr, y_tr = to_float(xq[tr]), y[tr]
x_va, y_va = to_float(xq[va]), y[va]
print(f"train {len(tr)}  val {len(va)}  test {len(te)}   (test set is NOT touched here)")

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(C.LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cw = None
if C.USE_CLASS_WEIGHTS:
    w = compute_class_weight("balanced", classes=np.arange(5), y=y_tr)
    cw = {i: float(v) for i, v in enumerate(w)}
    print("class weights:", cw)

callbacks = [
    keras.callbacks.ModelCheckpoint(str(C.OUT_DIR / "best_model.keras"), monitor="val_accuracy",
                                    save_best_only=True, verbose=1),
    keras.callbacks.CSVLogger(str(C.OUT_DIR / "training_log.csv")),
]

hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=C.EPOCHS,
                 batch_size=C.BATCH, shuffle=True, class_weight=cw, callbacks=callbacks, verbose=2)
model.save(C.OUT_DIR / "last_model.keras")

# ---- learning curves (paper Fig. 12)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(C.OUT_DIR / "learning_curves.png", dpi=130)
json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()},
          open(C.OUT_DIR / "history.json", "w"))
print("best val_accuracy: %.4f" % max(hist.history["val_accuracy"]))
