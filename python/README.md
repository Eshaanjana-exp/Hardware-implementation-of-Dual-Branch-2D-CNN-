# ECG arrhythmia PQP-CNN - Python side (data -> train -> test -> weights -> INT4)

Reproduces the software half of Mangaraj, Mahapatra & Ari (BSPC 109, 2025, 108063):
64x64 grayscale beat images from MIT-BIH, dual-branch 2D-CNN with **35,529 trainable parameters**, 5 AAMI classes.

## Run order
```
pip install -r requirements.txt
python 01_prepare_data.py      # extracts data/<mit-bih zip> (or downloads it), filters, cuts beats, makes 64x64 images
python 02_train.py             # float32 training (GPU recommended)                          -> outputs/best_model.keras
python 03_test.py              # float accuracy, per-class report, confusion matrix
python 04_extract_params.py    # every layer's kernel / bias / #params                       -> outputs/params_float/
python 05_quantize_int4.py     # paper-style quantisation + integer inference + .mem export  -> outputs/int4/
python 06_qat_finetune.py      # optional: quantisation-aware fine-tuning for all-4-bit activations
```
If you already trained a model you do NOT need to redo steps 1-4: copy the new `quant_core.py`, `05_quantize_int4.py`
and `06_qat_finetune.py` next to your files and run 05 / 06.

## Quantisation (what changed and why)
The first version of step 05 used signed symmetric INT4 weights, a static 4-bit activation range and power-of-two shifts.
That is NOT what the paper describes and it was far too lossy (77 % on the test set). Now (`quant_core.py`):

| | paper | step 05 now |
|---|---|---|
| scheme | uniform affine, layer-wise, S = (max-min)/15, Z = -round(min/S) (Eq. 1-4) | same |
| pixels | UINT4 | UINT4 (`round(x/255*15)`) |
| weights | UINT4 + zero-point | UINT4 + zero-point per layer (`W_MODE=asym`) |
| MAC result | wide integer, scaled back with a layer-wise scale factor SF (Fig. 9), clamp replaces ReLU | `y = clamp((acc*m0 + 2^(n-1)) >> n, 0, 2^ACT_BITS-1)`, `m0/2^n = s_in*s_w/s_out` |
| activation width | "INT8 for intermediate MAC results" (Sec. 4.3.3) but "UINT4 throughout" elsewhere | `ACT_BITS` = 8 / 6 / 4 |
| bias | added after scaling | added inside the accumulator (int32, at scale s_in*s_w) - more accurate, same idea |

Measured on the float model from the first run (test set 7,306 beats, float accuracy 97.26 %):

| configuration | integer accuracy |
|---|---|
| first version (signed INT4, pow-2 shift, 4-bit act) | 77.3 % |
| **paper scheme**, UINT4 weights/pixels, 8-bit activations | **96.93 %** (paper: 96.89 %) |
| paper scheme, 6-bit activations | 96.69 % |
| paper scheme, everything 4-bit, post-training | 93.12 % |
| paper scheme, everything 4-bit, after QAT (`06`) | **96.48 %** |
| signed INT4 weights (`W_MODE=sym`, no zero-point, matches an INT4-signed RTL), everything 4-bit, after QAT | **96.29 %** |

So the paper's 96.89 % is what you get with 4-bit weights and pixels and wider (8-bit) feature maps. If your RTL keeps every
activation in 4 bit, use step 06 and then `MODEL=qat_model.keras SWEEP=0 python 05_quantize_int4.py`.
For signed INT4 weights: `W_MODE=sym python 06_qat_finetune.py` then `W_MODE=sym MODEL=qat_model.keras SWEEP=0 python 05_quantize_int4.py`.

## What the RTL has to do differently with this scheme
* weights are stored as UINT4 (0..15) with one zero-point Z per layer (in `quant_config.json`); the MAC is
  `sum(x * w_q) - Z * sum(x)`. The pixel sum `sum(x)` over the window is the same for every filter, so one shared adder
  tree serves all N2 filters. (Equivalent: store `w_q - Z` in 5 bits.)
* SZW skips weights equal to Z (real zero), not weights equal to 0.
* requantisation is a small multiply + shift per layer (`sf` in the json), not a plain shift. Values are in the report table.
* adder = saturating add of the two branches, clamped to the activation maximum; max-pool 2x2; Flatten order is (H, W, C).
* `.mem` weight order `[C_out][C_in][kh][kw]`, one 4-bit hex nibble per line; biases are 32-bit hex; golden tensors `[C][H][W]`.

## Other things to know
1. The split is random over beats of all patients (as in the paper), so accuracy is optimistic; `outputs/ecg_images.npz`
   stores the record name of every beat (`rec`) if you want a patient-wise split.
2. Records 102 and 104 have no lead II. `LEAD_FALLBACK=True` uses their first channel (V5) so the paced-beat count is
   near the paper's (Q ~ 8,000 vs 7,772). Set it to False to skip them.
3. Default image polarity is black waveform on white paper; flip `DARK_ON_LIGHT` in `config.py` (retrain if you do).
