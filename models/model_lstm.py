"""CNN-LSTM image captioning model."""

import tensorflow as tf
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_lstm_model(vocab_size: int, max_len: int = config.MAX_CAPTION_LENGTH) -> tf.keras.Model:
    """
    Merge model:
      image_input  (2048,)  → Dense → RepeatVector → …
      text_input   (max_len,) → Embedding → LSTM → …
      merged → LSTM → Dense(vocab_size, softmax)
    """
    # --- Image branch ---
    img_input = tf.keras.layers.Input(shape=(config.FEATURE_SIZE,), name="image_input")
    img_dense = tf.keras.layers.Dense(config.EMBEDDING_DIM, activation="relu", name="img_dense")(img_input)
    img_dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="img_dropout")(img_dense)
    img_repeat = tf.keras.layers.RepeatVector(max_len, name="img_repeat")(img_dropout)

    # --- Text branch ---
    text_input = tf.keras.layers.Input(shape=(max_len,), name="text_input")
    text_embed = tf.keras.layers.Embedding(vocab_size, config.EMBEDDING_DIM, mask_zero=True, name="text_embed")(text_input)
    text_dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="text_dropout")(text_embed)

    # --- Merge ---
    merged = tf.keras.layers.Add(name="merge")([img_repeat, text_dropout])
    lstm_out = tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=False, name="lstm")(merged)
    lstm_dropout = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="lstm_dropout")(lstm_out)

    # --- Output ---
    output = tf.keras.layers.Dense(vocab_size, activation="softmax", name="output")(lstm_dropout)

    model = tf.keras.Model(inputs=[img_input, text_input], outputs=output, name="CNN_LSTM_Captioner")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
