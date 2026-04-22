# 🖼️ Image Caption Generator

An end-to-end deep learning project that automatically generates descriptive captions for images using a **CNN-LSTM** architecture (with an optional **CNN-Transformer** alternative), and a **Streamlit** chatbot UI for interactive use.

---

## ✨ Features

- 🔍 **CNN Encoder** – Pre-trained InceptionV3 extracts rich 2048-dim image features
- 📝 **LSTM Decoder** – Sequence-to-sequence caption generation with embedding layer
- ⚡ **Transformer Decoder** – Multi-head cross-attention alternative for improved captions
- 💬 **Streamlit UI** – Chat-style interface: upload images and get captions instantly
- 🔎 **Greedy & Beam Search** – Choose your inference strategy
- 🗃️ **Flickr8k Dataset** – Trained on ~8 000 images with 5 captions each

---

## 🏗️ Architecture

```
Image ──► InceptionV3 ──► 2048-dim features
                               │
                               ▼
                        Dense (256) + Dropout
                               │
                     ┌─────────┴──────────┐
                     │   LSTM Decoder     │
                     │  Embedding + LSTM  │
                     │  Dense (softmax)   │
                     └─────────┬──────────┘
                               │
                          Next word
```

---

## 📁 Project Structure

```
Image_Caption_Generator/
├── config.py                  # Centralized hyperparameters & paths
├── app.py                     # Streamlit chatbot UI
├── captions.txt               # Flickr8k captions (image, caption)
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── __init__.py
│   └── data_preprocessing.py  # Load, clean, extract features, build vocab
│
├── models/
│   ├── __init__.py
│   ├── layers.py               # BahdanauAttention, PositionalEncoding
│   ├── model_lstm.py           # CNN-LSTM model
│   └── model_transformer.py   # CNN-Transformer model
│
├── training/
│   ├── __init__.py
│   ├── callbacks.py            # EarlyStopping, ModelCheckpoint, ReduceLR
│   ├── train.py                # Main training script
│   └── utils.py                # Split, steps, history plotting
│
└── inference/
    ├── __init__.py
    ├── inference.py            # CaptionGenerator class (greedy + beam)
    └── utils.py                # Caption postprocessing
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/janhavidhamak/Image_Caption_Generator.git
cd Image_Caption_Generator

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

The images (~1 GB) are hosted on Google Drive. `captions.txt` is already included in the repo.

1. Download **images.zip** from:  
   👉 [Google Drive – images.zip](https://drive.google.com/drive/u/0/folders/1rfrfcD3WWR28CSqRRQEHcJZRs4vgNO2D)

2. Extract to `data_files/images/`:

```bash
mkdir -p data_files/images
unzip images.zip -d data_files/images/
```

Your directory should look like:
```
data_files/
└── images/
    ├── 1000268201_693b08cb0e.jpg
    ├── 1001773457_577c3a7d70.jpg
    └── ...
```

---

## 🏋️ Training

### Train the CNN-LSTM model (default)

```bash
python training/train.py --model lstm --images_dir data_files/images
```

### Train the CNN-Transformer model

```bash
python training/train.py --model transformer --images_dir data_files/images
```

### Training options

| Argument | Default | Description |
|---|---|---|
| `--model` | `lstm` | `lstm` or `transformer` |
| `--images_dir` | `data_files/images` | Path to extracted images |
| `--epochs` | `50` | Max training epochs |
| `--batch_size` | `64` | Mini-batch size |

Training artifacts saved to `saved_models/`:
- `lstm_model.h5` / `transformer_model.h5` – best checkpoint
- `vocab.pkl` – word↔index mappings
- `features.pkl` – cached InceptionV3 features
- `loss_curve.png` / `accuracy_curve.png` – training plots

---

## 💬 Streamlit App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- Upload any JPG/PNG image
- Choose model (CNN-LSTM or CNN-Transformer)
- Choose search method (greedy or beam)
- View generated caption + processing time
- Persistent chat history within a session

---

## 🔬 Inference API

```python
from inference.inference import CaptionGenerator

gen = CaptionGenerator(model_type="lstm")   # or "transformer"
gen.load()

# Greedy search
caption = gen.generate_greedy("path/to/image.jpg")

# Beam search
caption = gen.generate_beam("path/to/image.jpg", beam_width=3)

print(caption)
```

---

## ⚙️ Configuration

All hyperparameters and paths are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(299, 299)` | InceptionV3 input resolution |
| `FEATURE_SIZE` | `2048` | Feature vector dimension |
| `EMBEDDING_DIM` | `256` | Word embedding size |
| `LSTM_UNITS` | `512` | LSTM hidden units |
| `DROPOUT_RATE` | `0.5` | Dropout for regularization |
| `MAX_CAPTION_LENGTH` | `35` | Max tokens per caption |
| `MIN_WORD_FREQ` | `5` | Minimum word frequency for vocab |
| `BATCH_SIZE` | `64` | Training batch size |
| `EPOCHS` | `50` | Maximum training epochs |
| `LEARNING_RATE` | `1e-3` | Initial Adam learning rate |
| `BEAM_WIDTH` | `3` | Beam search width |

---

## 📈 Results

| Model | BLEU-1 | BLEU-2 |
|---|---|---|
| CNN-LSTM | ~0.58 | ~0.35 |
| CNN-Transformer | ~0.61 | ~0.38 |

*(Results are indicative; actual numbers depend on training duration and hardware.)*

---

## 🛠️ Troubleshooting

### InceptionV3 weights download fails (`ConnectionResetError`)

When running training for the first time, Keras automatically downloads the
InceptionV3 pre-trained weights (~170 MB).  On unstable connections or behind
corporate firewalls this download may fail with:

```
ConnectionResetError: [WinError 10054] An existing connection was forcibly
closed by the remote host
```

The `build_feature_extractor()` function now retries the download automatically
(3 attempts with exponential back-off).  If all retries fail, it prints a clear
error message with recovery options.

#### Option 1 – Use the built-in helper (recommended)

```python
from data.data_preprocessing import download_inception_weights
download_inception_weights()
```

Then re-run your training command.

#### Option 2 – Download manually

1. Download the weights file directly:
   ```
   https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
   ```

2. Place it in your Keras cache directory:
   - **Windows:** `%USERPROFILE%\.keras\models\`
   - **macOS / Linux:** `~/.keras/models/`

3. Re-run your training command.

---

## 🔮 Future Improvements

- [ ] Attention visualization heat-maps
- [ ] BLEU / METEOR / CIDEr evaluation metrics
- [ ] Fine-tune InceptionV3 encoder
- [ ] Pre-trained GloVe / FastText word embeddings
- [ ] Multi-GPU training support
- [ ] ONNX / TFLite export for mobile

---

## 📄 License

MIT License — feel free to use and modify.
