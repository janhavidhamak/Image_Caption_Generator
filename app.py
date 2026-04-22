"""Streamlit chatbot interface for the Image Caption Generator."""

import io
import os
import sys
import time
import logging

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .user-message { background-color: #e8f4fd; }
    .bot-message  { background-color: #f0f4e8; }
    .caption-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 0.25rem;
    }
    .info-text { font-size: 0.85rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Lazy model loading ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model… (first run may take a minute)")
def load_generator(model_type: str):
    """Load the CaptionGenerator (cached across reruns)."""
    from inference.inference import CaptionGenerator  # lazy import
    gen = CaptionGenerator(model_type=model_type)
    gen.load()
    return gen


def model_available(model_type: str) -> bool:
    path = config.LSTM_MODEL_PATH if model_type == "lstm" else config.TRANSFORMER_MODEL_PATH
    return os.path.exists(path) and os.path.exists(config.VOCAB_FILE)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/image-caption.png",
        width=80,
    )
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Model architecture",
        options=["lstm", "transformer"],
        format_func=lambda x: "CNN-LSTM" if x == "lstm" else "CNN-Transformer",
        help="Choose which trained model to use for inference.",
    )

    search_method = st.radio(
        "Caption search method",
        options=["greedy", "beam"],
        help="Beam search usually produces better captions but is slower.",
    )

    if search_method == "beam":
        beam_width = st.slider("Beam width", min_value=2, max_value=10, value=config.BEAM_WIDTH)
    else:
        beam_width = 1

    st.divider()
    st.markdown("### 📊 Model Info")
    st.markdown(
        f"""
        - **Encoder**: InceptionV3 (frozen)
        - **Feature size**: {config.FEATURE_SIZE}
        - **Embedding dim**: {config.EMBEDDING_DIM}
        - **LSTM units**: {config.LSTM_UNITS}
        - **Vocab size**: *(loaded at runtime)*
        - **Max caption length**: {config.MAX_CAPTION_LENGTH}
        """
    )

    st.divider()
    st.markdown("### 📁 Dataset")
    st.markdown(
        f"[Download images.zip]({config.GDRIVE_LINK})",
        unsafe_allow_html=False,
    )

    if not model_available(model_choice):
        st.warning(
            "⚠️ No trained model found. "
            "Please run `python training/train.py` first."
        )


# ── Session state ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # list of {"role": "user"|"bot", "content": ...}


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image and the model will generate a descriptive caption for it.")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image(msg["image"], use_container_width=True)
            with col2:
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<div><b>You</b><br><span class="info-text">{msg["filename"]}</span></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="chat-message bot-message">'
                f'<div>🤖 <b>Caption Bot</b><br>'
                f'<span class="caption-text">💬 {msg["caption"]}</span><br>'
                f'<span class="info-text">⏱ {msg["elapsed"]:.2f}s | 🔍 {msg["method"]} search | 🧠 {msg["model"]}</span>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

st.divider()

# Upload widget
uploaded = st.file_uploader(
    "📤 Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded is not None:
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    col_img, col_btn = st.columns([3, 1])
    with col_img:
        st.image(pil_img, caption=uploaded.name, use_container_width=True)
    with col_btn:
        generate_btn = st.button("✨ Generate Caption", use_container_width=True, type="primary")

    if generate_btn:
        if not model_available(model_choice):
            st.error("Model not found. Please train the model first (see README).")
        else:
            with st.spinner("Generating caption…"):
                try:
                    gen = load_generator(model_choice)
                    # Prepare image array for inference
                    from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
                    img_resized = pil_img.resize(config.IMAGE_SIZE)
                    img_arr = np.array(img_resized, dtype=np.float32)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    img_arr = inc_preprocess(img_arr)

                    t0 = time.time()
                    if search_method == "beam":
                        caption = gen._decode_beam(
                            gen._extract_feature_from_array(img_arr), beam_width=beam_width
                        )
                    else:
                        caption = gen._decode_greedy(gen._extract_feature_from_array(img_arr))
                    elapsed = time.time() - t0

                    # Append to chat history
                    st.session_state["chat_history"].append(
                        {
                            "role": "user",
                            "image": pil_img,
                            "filename": uploaded.name,
                        }
                    )
                    st.session_state["chat_history"].append(
                        {
                            "role": "bot",
                            "caption": caption,
                            "elapsed": elapsed,
                            "method": search_method,
                            "model": "CNN-LSTM" if model_choice == "lstm" else "CNN-Transformer",
                        }
                    )
                    st.success(f"✅ Caption: **{caption}**")
                    st.rerun()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                except (RuntimeError, ValueError) as exc:
                    st.error(f"Error generating caption: {exc}")
                    logger.exception("Caption generation failed")

if st.session_state["chat_history"]:
    if st.button("🗑️ Clear chat history"):
        st.session_state["chat_history"] = []
        st.rerun()
