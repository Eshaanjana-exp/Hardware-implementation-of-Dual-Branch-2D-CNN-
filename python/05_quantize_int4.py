"""
05_quantize_int4.py  -  float model -> integer model, following the paper's quantisation (see quant_core.py)

    pixels   UINT4 (0..15)
    weights  UINT4 with a per-layer zero-point (asymmetric affine, paper Eq. 1-4)      [W_MODE=asym, default]
             (W_MODE=sym gives signed INT4 -8..7 without zero-point, noticeably less accurate)
    MAC      acc = sum x*(w_q - Z) + bias_int32,   y = clamp((acc*m0 + 2^(n-1)) >> n, 0, 2^ACT_BITS - 1)

Environment switches
    MODEL     model file in outputs/         default best_model.keras   (use qat_model.keras after 06_qat_finetune.py)
    ACT_BITS  activation width that is exported   default 4  (8 reproduces the paper's number, 4 = strict UINT4)
    W_MODE    asym | sym                     default asym
    CALIB_PCT calibration percentile         default depends on ACT_BITS
    SWEEP     1 = also print accuracy for ACT_BITS 8 / 6 / 4 before exporting      default 1

Outputs in outputs/int4/:  <layer>_weights.mem, <layer>_bias.mem, quant_config.json, weights_int_all.npz, golden/
"""
import json, os
import numpy as np
from tensorflow import keras
import config as C
from data_utils import load_xy, split_indices, to_float
import quant_core as QC

MODEL = os.environ.get("MODEL", "best_model.keras")
ACT_BITS = int(os.environ.get("ACT_BITS", 4))
W_MODE = os.environ.get("W_MODE", "asym")
SWEEP = os.environ.get("SWEEP", "1") == "1"
CALIB_N = 1000


def hexes(a, bits):
    d = (bits + 3) // 4
    m = (1 << bits) - 1
    return [f"{int(v) & m:0{d}X}" for v in np.asarray(a).ravel()]


def wr(path, lines):
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    model = keras.models.load_model(C.OUT_DIR / MODEL, compile=False)
    xq, y = load_xy()
    tr, _, te = split_indices(y)
    x_cal = to_float(xq[tr[:CALIB_N]])
    x_te, y_te = xq[te], y[te]

    pf = model.predict(to_float(x_te), batch_size=256, verbose=0).argmax(1)
    acc_f = (pf == y_te).mean()
    print(f"model {MODEL}   weights: {W_MODE}   un-quantised (float) test accuracy of this file: {acc_f * 100:.2f} %  ({len(te)} beats)\n")

    scales_file = C.OUT_DIR / (MODEL.replace(".keras", "") + ".scales.json")     # written by 06_qat_finetune.py
    fixed = json.load(open(scales_file)) if scales_file.exists() else None

    def make_scales(bits):
        if fixed is not None and fixed["act_bits"] == bits:
            return fixed["scales"]
        pct = float(os.environ.get("CALIB_PCT", QC.DEFAULT_PCT.get(bits, 99.9)))
        return QC.calibrate(model, x_cal, bits, pct)

    if SWEEP:
        print("ACT_BITS sweep (weights + pixels are always 4-bit):")
        for bits in (8, 6, 4):
            Qb = QC.quantise_model(model, make_scales(bits), W_MODE)
            a = (QC.predict_int(x_te, Qb, bits).argmax(1) == y_te).mean()
            print(f"   activations {bits}-bit : integer accuracy {a * 100:.2f} %")
        print()

    # ---------------------------------------------------------------- the exported configuration
    S = make_scales(ACT_BITS)
    Q = QC.quantise_model(model, S, W_MODE)
    stats = {}
    logits = QC.predict_int(x_te, Q, ACT_BITS, stats=stats)
    pi = logits.argmax(1)
    acc_i = (pi == y_te).mean()
    print(f"EXPORT: weights {W_MODE} 4-bit, activations {ACT_BITS}-bit\n"
          f"Test accuracy   float model file: {acc_f * 100:.2f} %    integer model: {acc_i * 100:.2f} %   "
          f"(paper: 97.79 -> 96.89)    float/int agreement: {(pf == pi).mean() * 100:.2f} %")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_te, pi, target_names=C.CLASSES, digits=4, zero_division=0))
    print("confusion matrix (rows true, cols predicted):\n", confusion_matrix(y_te, pi))

    # ---------------------------------------------------------------- per-layer report
    probe = keras.Model(model.input, [model.get_layer(n).output for n, _, _ in QC.LAYERS[:-1]])
    n_chk = min(200, len(te))
    fl = probe.predict(to_float(x_te[:n_chk]), verbose=0)
    _, T = QC.int_forward(x_te[:n_chk, ..., None].astype(np.int64), Q, ACT_BITS)
    print("\n%-8s %-4s %-9s %-16s %-9s %-10s %-9s %s" %
          ("layer", "Z_w", "s_w", "SF = m0 >> n", "zero_w %", "|acc|max", "acc bits", "mean|float-int| (LSB)"))
    cfg = {"model": MODEL, "weight_mode": W_MODE, "act_bits": ACT_BITS, "scales": S,
           "acc_float": float(acc_f), "acc_int": float(acc_i), "layers": {}}
    for (name, kin, kout), fo in zip(QC.LAYERS[:-1], fl):
        q = Q[name]
        zero = 100.0 * float((q["W_eff"] == 0).mean())                      # weights equal to real zero -> skipped by SZW
        bits = int(stats[name]).bit_length() + 1
        err = float(np.abs(fo / S[kout] - T[name]).mean())
        print("%-8s %-4d %-9.5f %-16s %-9.1f %-10d %-9d %.3f" %
              (name, q["Z"], q["s_w"], f"{q['rq']['mult']} >> {q['rq']['shift']}", zero, stats[name], bits, err))
        cfg["layers"][name] = dict(zero_point=q["Z"], s_w=q["s_w"], s_in=q["s_in"], s_out=q["s_out"], sf=q["rq"],
                                   zero_weight_pct=zero, max_abs_acc=int(stats[name]), acc_bits=bits)
    q = Q["dense_2"]
    bits = int(stats["dense_2"]).bit_length() + 1
    print("%-8s %-4d %-9.5f %-16s %-9.1f %-10d %-9d (logits, arg-max only)" %
          ("dense_2", q["Z"], q["s_w"], "-", 100.0 * float((q["W_eff"] == 0).mean()), stats["dense_2"], bits))
    cfg["layers"]["dense_2"] = dict(zero_point=q["Z"], s_w=q["s_w"], s_in=q["s_in"], sf=None,
                                    max_abs_acc=int(stats["dense_2"]), acc_bits=bits)

    # ---------------------------------------------------------------- export
    odir = C.OUT_DIR / "int4"
    (odir / "golden").mkdir(parents=True, exist_ok=True)
    npz = {}
    for name, _, _ in QC.LAYERS:
        q = Q[name]
        Wq = q["Wq"]
        w_hw = np.transpose(Wq, (3, 2, 0, 1)) if Wq.ndim == 4 else Wq.T        # [C_out][C_in][kh][kw]
        wr(odir / f"{name}_weights.mem", hexes(w_hw, 4))                        # asym: 0..15   sym: two's complement
        wr(odir / f"{name}_bias.mem", hexes(q["bq"], 32))
        npz[f"{name}/Wq"] = Wq; npz[f"{name}/Z"] = q["Z"]; npz[f"{name}/bq"] = q["bq"]
        if q["rq"]:
            npz[f"{name}/sf_mult"] = q["rq"]["mult"]; npz[f"{name}/sf_shift"] = q["rq"]["shift"]
    np.savez(odir / "weights_int_all.npz", **npz)
    json.dump(cfg, open(odir / "quant_config.json", "w"), indent=1)

    _, T0 = QC.int_forward(x_te[:1, ..., None].astype(np.int64), Q, ACT_BITS)     # golden vectors for test image #0
    for k, v in T0.items():
        if k == "logits":
            np.save(odir / "golden" / "logits.npy", v[0])
            continue
        t = v[0]
        chw = np.transpose(t, (2, 0, 1)) if t.ndim == 3 else t                 # [C][H][W]
        np.save(odir / "golden" / f"{k}.npy", t)
        wr(odir / "golden" / f"{k}.mem", hexes(chw, 4 if k == "input" else ACT_BITS))
    print("\ntest image #0: label", C.CLASSES[y_te[0]], "| integer prediction", C.CLASSES[int(T0["logits"][0].argmax())])
    print("Exported to", odir)


if __name__ == "__main__":
    main()
