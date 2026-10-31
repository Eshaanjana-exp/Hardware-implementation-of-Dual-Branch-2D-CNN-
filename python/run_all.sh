#!/usr/bin/env bash
# Whole flow. Use a GPU machine (e.g. Google Colab) for step 2.
set -e
python 01_prepare_data.py      # MIT-BIH -> 64x64 images                       (put the dataset .zip in data/ first)
python 02_train.py             # float32 training                              -> outputs/best_model.keras
python 03_test.py              # float test metrics
python 04_extract_params.py    # weights / biases / parameter counts (float)
python 05_quantize_int4.py     # paper-style quantisation, integer accuracy, .mem export (prints an ACT_BITS sweep)
# optional - only if you need ALL activations in 4 bit and 05 shows a large drop:
python 06_qat_finetune.py
MODEL=qat_model.keras python 05_quantize_int4.py
