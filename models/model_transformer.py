"""Transformer-based image captioning model (alternative to LSTM)."""

import tensorflow as tf
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.layers import PositionalEncoding


def _multi_head_attention_block(d_model, num_heads, dff, dropout_rate):
    """Return a single transformer decoder block as a Keras functional sub-graph."""
    # inputs
    query = tf.keras.layers.Input(shape=(None, d_model))
    key_value = tf.keras.layers.Input(shape=(None, d_model))

    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query, key_value)
    attn = tf.keras.layers.Dropout(dropout_rate)(attn)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(query + attn)

    ffn = tf.keras.layers.Dense(dff, activation="relu")(out1)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return tf.keras.Model(inputs=[query, key_value], outputs=out2)


def build_transformer_model(vocab_size: int, max_len: int = config.MAX_CAPTION_LENGTH) -> tf.keras.Model:
    """
    Image features projected to d_model, then used as encoder memory.
    Caption tokens go through embedding + positional encoding then cross-attend to image features.
    """
    d_model = config.TRANSFORMER_D_MODEL
    num_heads = config.TRANSFORMER_NUM_HEADS
    dff = config.TRANSFORMER_DFF
    num_layers = config.TRANSFORMER_NUM_LAYERS
    dropout = config.TRANSFORMER_DROPOUT

    # --- Encoder side (image) ---
    img_input = tf.keras.layers.Input(shape=(config.FEATURE_SIZE,), name="image_input")
    img_proj = tf.keras.layers.Dense(d_model, activation="relu", name="img_proj")(img_input)
    img_proj = tf.keras.layers.Reshape((1, d_model), name="img_reshape")(img_proj)  # (batch, 1, d_model)

    # --- Decoder side (text) ---
    text_input = tf.keras.layers.Input(shape=(max_len,), name="text_input")
    text_embed = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True, name="text_embed")(text_input)
    text_embed = PositionalEncoding(max_len, d_model, name="pos_enc")(text_embed)
    text_embed = tf.keras.layers.Dropout(dropout, name="embed_drop")(text_embed)

    x = text_embed
    for i in range(num_layers):
        # self-attention
        self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"self_attn_{i}"
        )(x, x)
        self_attn = tf.keras.layers.Dropout(dropout)(self_attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x + self_attn)

        # cross-attention over image features
        cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"cross_attn_{i}"
        )(x, img_proj)
        cross_attn = tf.keras.layers.Dropout(dropout)(cross_attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x + cross_attn)

        # feed-forward
        ffn = tf.keras.layers.Dense(dff, activation="relu", name=f"ffn1_{i}")(x)
        ffn = tf.keras.layers.Dense(d_model, name=f"ffn2_{i}")(ffn)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln3_{i}")(x + ffn)

    # --- Output ---
    output = tf.keras.layers.Dense(vocab_size, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=[img_input, text_input], outputs=output, name="CNN_Transformer_Captioner")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
