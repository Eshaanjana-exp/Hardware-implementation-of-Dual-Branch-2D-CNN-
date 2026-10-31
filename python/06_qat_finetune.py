"""
06_qat_finetune.py  -  quantisation-aware fine-tuning (only needed if you want ALL-4-bit activations)

Post-training quantisation with 4-bit activations loses about 4 points (97.3 -> 93). This script continues training
the float model for a few epochs while the forward pass already uses the integer datapath of 05_quantize_int4.py
(UINT4 asymmetric weights, clamped + rounded activations); the backward pass uses the straight-through estimator.

    python 06_qat_finetune.py                 # ACT_BITS=4, W_MODE=asym, 10 epochs
    MODEL=qat_model.keras python 05_quantize_int4.py     # integer accuracy + .mem export of the fine-tuned model

Environment: ACT_BITS (4), W_MODE (asym), EPOCHS (10), LR (5e-5), CALIB_PCT
Writes outputs/qat_model.keras (same architecture, so 03/04 work on it too) and outputs/qat_model.scales.json
"""
import json, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config as C
from data_utils import load_xy, split_indices, to_float
import quant_core as QC

ACT_BITS = int(os.environ.get("ACT_BITS", 4))
W_MODE = os.environ.get("W_MODE", "asym")
EPOCHS = int(os.environ.get("EPOCHS", 10))
LR = float(os.environ.get("LR", 5e-5))
BATCH = C.BATCH
keras.utils.set_random_seed(C.SEED)

model = keras.models.load_model(C.OUT_DIR / "best_model.keras", compile=False)
xq, y = load_xy()
tr, va, te = split_indices(y)
x_tr, y_tr = to_float(xq[tr]), y[tr]
x_va, y_va = to_float(xq[va]), y[va]

pct = float(os.environ.get("CALIB_PCT", QC.DEFAULT_PCT.get(ACT_BITS, 99.9)))
S = QC.calibrate(model, x_tr[:1000], ACT_BITS, pct)          # activation scales stay fixed during fine-tuning
print("activation scales:", {k: round(v, 4) for k, v in S.items()})

V = {n: (tf.Variable(model.get_layer(n).get_weights()[0]), tf.Variable(model.get_layer(n).get_weights()[1]))
     for n in QC.NAMES}
train_vars = [v for kb in V.values() for v in kb]
opt = keras.optimizers.Adam(LR)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        loss = loss_fn(yb, QC.qat_forward(xb, V, S, W_MODE, ACT_BITS))
    opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
    return loss


def evaluate(x, yy):
    p = [tf.argmax(QC.qat_forward(tf.constant(x[i:i + 512]), V, S, W_MODE, ACT_BITS), 1).numpy()
         for i in range(0, len(x), 512)]
    return float((np.concatenate(p) == yy).mean())


def save(path):
    m = keras.models.load_model(C.OUT_DIR / "best_model.keras", compile=False)   # same architecture
    for n in QC.NAMES:
        m.get_layer(n).set_weights([V[n][0].numpy(), V[n][1].numpy()])
    m.save(path)


best = evaluate(x_va, y_va)
print(f"before fine-tuning: fake-quant validation accuracy {best * 100:.2f} %")
save(C.OUT_DIR / "qat_model.keras")
json.dump({"act_bits": ACT_BITS, "weight_mode": W_MODE, "scales": S}, open(C.OUT_DIR / "qat_model.scales.json", "w"))

ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(len(x_tr), seed=C.SEED, reshuffle_each_iteration=True).batch(BATCH)
for ep in range(1, EPOCHS + 1):
    losses = [float(train_step(xb, yb)) for xb, yb in ds]
    acc = evaluate(x_va, y_va)
    flag = ""
    if acc > best:
        best = acc
        save(C.OUT_DIR / "qat_model.keras")
        flag = "  <- saved"
    print(f"epoch {ep:2d}/{EPOCHS}  loss {np.mean(losses):.4f}  val acc (fake-quant) {acc * 100:.2f} %{flag}", flush=True)

print(f"\nbest validation accuracy {best * 100:.2f} %   ->  now run:  MODEL=qat_model.keras python 05_quantize_int4.py")
