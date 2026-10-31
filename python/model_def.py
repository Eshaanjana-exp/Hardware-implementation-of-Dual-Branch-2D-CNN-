"""
model_def.py - the dual-branch 2D-CNN of the paper (Fig. 5 / Table 3), layer names identical to Table 3.

Run  `python model_def.py`  to print the summary; it must report 35,529 trainable parameters.
"""
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import config as C


def _conv(x, filters, k, name):
    # 'valid' padding, stride 1. ReLU here = the clamp-at-zero that the UINT4 requantiser does in hardware.
    return layers.Conv2D(filters, k, strides=1, padding="valid", activation="relu",
                         kernel_regularizer=regularizers.l2(C.L2), name=name)(x)


# per stage: (5x5 branch, first 3x3 of the 2-conv branch, second 3x3)
_NAMES = {1: ("conv_11", "conv_12", "conv_13"),
          2: ("conv_22", "conv_21", "conv_23"),
          3: ("conv_33", "conv_31", "conv_32")}


def build_model():
    inp = keras.Input(shape=(C.IMG, C.IMG, 1), name="input")
    x = inp
    for stage, filters in zip((1, 2, 3), (8, 16, 32)):
        n5, n3a, n3b = _NAMES[stage]
        branch_a = _conv(x, filters, 5, n5)                         # single 5x5 conv  (broad context)
        branch_b = _conv(_conv(x, filters, 3, n3a), filters, 3, n3b)  # two 3x3 convs   (fine detail)
        x = layers.Add(name=f"adder{stage}")([branch_a, branch_b])  # filter-wise add
        x = layers.MaxPooling2D(pool_size=2, name=f"pool_{stage}")(x)
    x = _conv(x, 16, 1, "conv_44")                                  # 1x1 conv, keeps flatten size at 64
    x = layers.MaxPooling2D(pool_size=2, name="pool_4")(x)
    x = layers.Flatten(name="flatten")(x)                           # order is (H, W, C)
    x = layers.Dense(10, activation="relu", kernel_regularizer=regularizers.l2(C.L2), name="dense_1")(x)
    out = layers.Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(C.L2), name="dense_2")(x)
    return keras.Model(inp, out, name="pqp_cnn_dual_branch")


if __name__ == "__main__":
    m = build_model()
    m.summary()
    n = m.count_params()
    print("\nTotal trainable parameters:", n, "(paper: 35,529)")
    assert n == 35529, "architecture does not match Table 3"
