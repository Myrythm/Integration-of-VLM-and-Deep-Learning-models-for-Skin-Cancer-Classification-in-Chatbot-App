# Image Classification Model

This directory holds the trained `EfficientNetB3` model used by the v2 chatbot for skin lesion classification.

## Model Details

- **Architecture**: EfficientNetB3 (frozen) → Dense(256) → Dropout → Dense(4, softmax)
- **Input size**: 224×224 RGB
- **Output**: 4-class softmax over `SKIN_CANCER_LABELS` (see `utils/image_classifier.py`)
  - `Karsinoma Sel Basal` (Basal Cell Carcinoma)
  - `Karsinoma Sel Skuamosa` (Squamous Cell Carcinoma)
  - `Melanoma`
  - `Nevus`
- **Total params**: ~11.2M (~42.66 MB on disk as H5)

## Setup

The model file (`skinCancer.h5`, ~134 MB) is **gitignored** and must be placed here before the app can classify images.

1. **Place the file** at `./model/skinCancer.h5` (relative to the project root), OR
2. **Override the path** via the `MODEL_PATH` env var in `.env`:
   ```env
   MODEL_PATH=/path/to/your/skinCancer.h5
   ```

## Verifying the Model

```bash
source .venv/bin/activate
python -c "from utils.image_classifier import _get_model; m = _get_model('./model/skinCancer.h5'); print('Output shape:', m.output_shape)"
```

Expected output: `Output shape: (None, 4)`

## TensorFlow Version

The model is compatible with TensorFlow 2.9+ (saved with TF 2.9.1). The v2 project uses **TensorFlow 2.15.0** on Python 3.10+ (recommended for current hardware/OS).

If you must use TF 2.9.1, use Python ≤ 3.10.9 to avoid ABI mismatch (TF 2.9.1 wheels are built for Python 3.7-3.10.9, not 3.10.20+).

## Retraining

To retrain, the model is saved as `.h5` and can be loaded with:
```python
import tensorflow as tf
model = tf.keras.models.load_model('./model/skinCancer.h5')
```

See the v1 legacy `utils/image_processing.py` for the preprocessing pipeline (PIL → RGB → resize 224×224 → numpy → expand_dims).
