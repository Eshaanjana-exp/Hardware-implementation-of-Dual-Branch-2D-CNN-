"""
04_extract_params.py  -  pull weights, biases and parameter counts out of every layer

Keras kernel layouts
    Conv2D kernel : (kh, kw, C_in, C_out)      bias : (C_out,)
    Dense  kernel : (n_in, n_out)              bias : (n_out,)
Hardware ("PQP") layout written to the .txt files:  [C_out][C_in][kh][kw]   (Dense: [n_out][n_in])
"""
import json
import numpy as np
from tensorflow import keras
import config as C

model = keras.models.load_model(C.OUT_DIR / "best_model.keras", compile=False)

# 1) the standard Keras summary
model.summary()

# 2) layers in the same order as Table 3 of the paper
ORDER = ["conv_11", "conv_12", "conv_13", "conv_22", "conv_21", "conv_23",
         "conv_33", "conv_31", "conv_32", "conv_44", "dense_1", "dense_2"]

pdir = C.OUT_DIR / "params_float"
pdir.mkdir(exist_ok=True)

rows, npz = [], {}
tot_k = tot_b = 0
print("\n%-8s %-14s %-16s %7s %6s %7s | %9s %9s %8s %7s" %
      ("layer", "kernel shape", "output shape", "#kern", "#bias", "total", "w_min", "w_max", "w_mean", "b_max"))
for name in ORDER:
    layer = model.get_layer(name)
    k, b = layer.get_weights()                      # <- this is all it takes
    assert len(layer.trainable_weights) == 2       # kernel + bias, nothing frozen
    tot_k += k.size; tot_b += b.size
    print("%-8s %-14s %-16s %7d %6d %7d | %9.4f %9.4f %8.4f %7.3f" %
          (name, str(k.shape), str(layer.output.shape[1:]), k.size, b.size, k.size + b.size,
           k.min(), k.max(), k.mean(), np.abs(b).max()))

    # hardware ordering
    k_hw = np.transpose(k, (3, 2, 0, 1)) if k.ndim == 4 else k.T      # conv: (out,in,kh,kw)  dense: (out,in)
    np.save(pdir / f"{name}_kernel.npy", k)                            # Keras layout
    np.save(pdir / f"{name}_bias.npy", b)
    np.savetxt(pdir / f"{name}_kernel_hw.txt", k_hw.reshape(-1), fmt="%.8f")   # one value per line
    np.savetxt(pdir / f"{name}_bias.txt", b, fmt="%.8f")
    npz[f"{name}/kernel"] = k; npz[f"{name}/bias"] = b
    rows.append(dict(layer=name, kernel_shape=list(k.shape), out_shape=[int(v) for v in layer.output.shape[1:]],
                     kernel_params=int(k.size), bias_params=int(b.size), total=int(k.size + b.size)))

np.savez(C.OUT_DIR / "weights_float_all.npz", **npz)
json.dump(rows, open(C.OUT_DIR / "params_table.json", "w"), indent=1)

total = tot_k + tot_b
print(f"\nkernel params {tot_k}  +  bias params {tot_b}  =  {total}")
print("model.count_params()          :", model.count_params())
print("sum over trainable_weights    :", int(sum(np.prod(w.shape) for w in model.trainable_weights)))
print("non-trainable                 :", int(sum(np.prod(w.shape) for w in model.non_trainable_weights)))
print("paper (Table 3)               : 35529")
assert total == model.count_params() == 35529

# 3) how the count is made:  conv = kh*kw*C_in*C_out + C_out      dense = n_in*n_out + n_out
print("\nExample  conv_11 : 5*5*1*8 + 8   =", 5 * 5 * 1 * 8 + 8)
print("Example  conv_22 : 5*5*8*16 + 16 =", 5 * 5 * 8 * 16 + 16)
print("Example  dense_1 : 64*10 + 10     =", 64 * 10 + 10)

# 4) look at one filter
k = np.load(pdir / "conv_11_kernel.npy")
print("\nconv_11, filter 0 (5x5, 1 input channel):\n", np.round(k[:, :, 0, 0], 4))
print("conv_11 biases:", np.round(np.load(pdir / "conv_11_bias.npy"), 4))
print("\nFiles written to", pdir)
