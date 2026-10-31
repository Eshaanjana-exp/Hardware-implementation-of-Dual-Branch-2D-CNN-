"""
quant_core.py - quantisation as described in the paper (Section 3.3.1, Eq. 1-4), shared by
05_quantize_int4.py (post-training) and 06_qat_finetune.py (quantisation-aware fine-tuning).

What the paper does
    * uniform AFFINE (asymmetric) quantisation, layer-wise:  S = (x_max - x_min) / (q_max - q_min)
                                                            Z = q_min - round(x_min / S)
                                                            x_q = clamp(round(x / S) + Z, 0, 15)
    * input pixels  : UINT4 (0..15)
    * weights       : UINT4 (0..15) + one zero-point Z per layer  ->  MAC uses (w_q - Z)
    * MAC result    : wide integer, then multiplied by a layer-wise scale factor SF (Fig. 9) and clamped
                      to the activation range.  The clamp at 0 is what replaces ReLU.
    * "intermediate MAC results are INT8" (Section 4.3.3): activation width is therefore a parameter here
      (ACT_BITS = 8 reproduces the paper's 96.89 %, ACT_BITS = 4 is the strict all-UINT4 datapath).

Integer model of one layer:
    acc = sum_k x_k * (w_q,k - Z)  +  b_int              (int32; b_int = round(b / (s_in * s_w)))
    y   = clamp( (acc * m0 + 2^(n-1)) >> n , 0 , 2^ACT_BITS - 1 )      with  m0 / 2^n  ~=  s_in * s_w / s_out
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

# (layer, input-scale key, output-scale key). A key is 'input' or the name of the layer/adder whose output scale it is.
LAYERS = [("conv_11", "input", "adder1"), ("conv_12", "input", "conv_12"), ("conv_13", "conv_12", "adder1"),
          ("conv_22", "adder1", "adder2"), ("conv_21", "adder1", "conv_21"), ("conv_23", "conv_21", "adder2"),
          ("conv_33", "adder2", "adder3"), ("conv_31", "adder2", "conv_31"), ("conv_32", "conv_31", "adder3"),
          ("conv_44", "adder3", "conv_44"), ("dense_1", "conv_44", "dense_1"), ("dense_2", "dense_1", None)]
NAMES = [n for n, _, _ in LAYERS]
SCALE_KEYS = ["conv_12", "adder1", "conv_21", "adder2", "conv_31", "adder3", "conv_44", "dense_1"]

# calibration percentile that worked best per activation width (ablation on the trained model)
DEFAULT_PCT = {8: 99.99, 6: 99.9, 4: 99.5}


def levels(bits):
    return 2 ** bits - 1


# ====================================================================== calibration
def calibrate(model, x_float, act_bits, pct):
    """Layer-wise activation scales:  s = percentile(float activation) / (2^bits - 1)."""
    probe = keras.Model(model.input, [model.get_layer(k).output for k in SCALE_KEYS])
    acts = probe.predict(x_float, batch_size=256, verbose=0)
    S = {"input": 1.0 / 15.0}                       # pixels were made as round(x/255*15) -> x_q/15
    for k, a in zip(SCALE_KEYS, acts):
        S[k] = max(float(np.percentile(a, pct)), 1e-6) / levels(act_bits)
    return S


# ====================================================================== weights (paper Eq. 1-4)
def quantise_weights(W, mode="asym", bits=4):
    """Returns (q, S, Z).   asym: q in 0..15, effective integer weight = q - Z   (paper)
                            sym : q in -8..7, Z = 0                              (signed INT4)"""
    L = 2 ** bits
    if mode == "asym":
        lo, hi = min(float(W.min()), 0.0), max(float(W.max()), 0.0)     # range must contain real zero
        S = (hi - lo) / (L - 1)                                         # Eq. (1)
        Z = int(round(-lo / S))                                         # Eq. (2)
        q = np.clip(np.round(W / S) + Z, 0, L - 1).astype(np.int64)     # Eq. (3)+(4)
        return q, S, Z
    S = float(np.abs(W).max()) / (L // 2 - 1)
    return np.clip(np.round(W / S), -L // 2, L // 2 - 1).astype(np.int64), S, 0


def quantise_layer(s_in, s_out, W, b, wmode):
    q, s_w, Z = quantise_weights(W, wmode)
    bq = np.clip(np.round(b / (s_in * s_w)), -2 ** 31, 2 ** 31 - 1).astype(np.int64)    # bias at accumulator scale
    rq = None
    if s_out is not None:
        M = s_in * s_w / s_out                                          # the layer-wise scale factor SF
        n = int(math.floor(math.log2((2 ** 15 - 1) / M)))               # 15-bit multiplier mantissa
        assert n >= 1, "scale factor too large for the fixed-point form"
        rq = {"mult": int(round(M * 2 ** n)), "shift": n, "M": float(M)}
    return dict(Wq=q, Z=Z, W_eff=(q - Z), bq=bq, s_w=float(s_w), s_in=float(s_in),
                s_out=None if s_out is None else float(s_out), rq=rq, wmode=wmode)


def quantise_model(model, S, wmode):
    Q = {}
    for name, kin, kout in LAYERS:
        W, b = model.get_layer(name).get_weights()
        Q[name] = quantise_layer(S[kin], S[kout] if kout else None, W, b, wmode)
    return Q


# ====================================================================== integer inference engine
def _conv_valid(x, W):
    # exact: every value is a small integer and stays far below 2^24, so float32 arithmetic has no rounding
    y = tf.nn.conv2d(x.astype(np.float32), W.astype(np.float32), 1, "VALID").numpy()
    return np.rint(y).astype(np.int64)


def _maxpool2(x):
    B, H, W, C = x.shape
    return x[:, :H // 2 * 2, :W // 2 * 2].reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def requant(acc, rq, lv):
    y = (acc * rq["mult"] + (1 << (rq["shift"] - 1))) >> rq["shift"]     # round-half-up
    return np.clip(y, 0, lv)


def int_forward(x_int, Q, act_bits, stats=None):
    """x_int (B,64,64,1) ints 0..15  ->  (int logits, dict of all intermediate integer tensors)."""
    lv = levels(act_bits)
    T = {"input": x_int}

    def conv(name, inp):
        q = Q[name]
        acc = _conv_valid(inp, q["W_eff"]) + q["bq"]
        if stats is not None:
            stats[name] = max(stats.get(name, 0), int(np.abs(acc).max()))
        T[name] = requant(acc, q["rq"], lv)
        return T[name]

    def stage(tag, x, n5, n3a, n3b):
        a = conv(n5, x)
        b = conv(n3b, conv(n3a, x))
        T[f"adder{tag}"] = np.clip(a + b, 0, lv)                          # saturating add of the two branches
        T[f"pool_{tag}"] = _maxpool2(T[f"adder{tag}"])
        return T[f"pool_{tag}"]

    x = stage(1, x_int, "conv_11", "conv_12", "conv_13")
    x = stage(2, x, "conv_22", "conv_21", "conv_23")
    x = stage(3, x, "conv_33", "conv_31", "conv_32")
    x = _maxpool2(conv("conv_44", x))
    T["pool_4"] = x
    flat = x.reshape(len(x), -1)                                          # (H, W, C) order = Keras Flatten
    q1, q2 = Q["dense_1"], Q["dense_2"]
    acc1 = flat @ q1["W_eff"] + q1["bq"]
    T["dense_1"] = requant(acc1, q1["rq"], lv)
    logits = T["dense_1"] @ q2["W_eff"] + q2["bq"]
    if stats is not None:
        stats["dense_1"] = max(stats.get("dense_1", 0), int(np.abs(acc1).max()))
        stats["dense_2"] = max(stats.get("dense_2", 0), int(np.abs(logits).max()))
    T["logits"] = logits
    return logits, T


def predict_int(xq, Q, act_bits, bs=256, stats=None):
    out = []
    for i in range(0, len(xq), bs):
        lg, _ = int_forward(xq[i:i + bs, ..., None].astype(np.int64), Q, act_bits, stats)
        out.append(lg)
    return np.concatenate(out)


# ====================================================================== fake-quant with straight-through estimator (for QAT)
def fq_weight_ste(W, mode):
    """Differentiable version of quantise_weights (paper Eq. 1-4). Gradient passes straight through the rounding."""
    if mode == "asym":
        lo = tf.minimum(tf.reduce_min(W), 0.0)
        hi = tf.maximum(tf.reduce_max(W), 0.0)
        S = tf.stop_gradient((hi - lo) / 15.0)
        Z = tf.stop_gradient(tf.round(-lo / S))
        q = tf.clip_by_value(tf.round(W / S) + Z, 0.0, 15.0)
        What = (q - Z) * S
    else:
        S = tf.stop_gradient(tf.reduce_max(tf.abs(W)) / 7.0)
        What = tf.clip_by_value(tf.round(W / S), -8.0, 7.0) * S
    return W + tf.stop_gradient(What - W)


def fq_act_ste(a, scale, lv):
    """Clamp to [0, lv*scale] and round to the grid; gradient is 1 inside the range and 0 outside."""
    a_c = tf.clip_by_value(a, 0.0, lv * scale)
    return a_c + tf.stop_gradient(tf.round(a_c / scale) * scale - a_c)


def qat_forward(x, V, S, wmode, act_bits):
    """Float forward pass that mimics the integer engine (quantised weights, clamped/rounded activations).
    V: {layer: (kernel_tensor, bias_tensor)}   S: activation scales."""
    lv = levels(act_bits)

    def conv(name, inp, out_key):
        k, b = V[name]
        o = tf.nn.relu(tf.nn.conv2d(inp, fq_weight_ste(k, wmode), 1, "VALID") + b)
        return fq_act_ste(o, S[out_key], lv)

    def stage(tag, inp, n5, n3a, n3b):
        key = f"adder{tag}"
        a = conv(n5, inp, key)
        b = conv(n3b, conv(n3a, inp, n3a), key)
        return tf.nn.max_pool2d(fq_act_ste(a + b, S[key], lv), 2, 2, "VALID")

    x = stage(1, x, "conv_11", "conv_12", "conv_13")
    x = stage(2, x, "conv_22", "conv_21", "conv_23")
    x = stage(3, x, "conv_33", "conv_31", "conv_32")
    x = tf.nn.max_pool2d(conv("conv_44", x, "conv_44"), 2, 2, "VALID")
    x = tf.reshape(x, (tf.shape(x)[0], -1))
    k, b = V["dense_1"]
    x = fq_act_ste(tf.nn.relu(x @ fq_weight_ste(k, wmode) + b), S["dense_1"], lv)
    k, b = V["dense_2"]
    return x @ fq_weight_ste(k, wmode) + b
