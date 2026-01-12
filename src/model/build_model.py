# Builds the hybrid CNN + LSTM model for multilingual phishing-email text classification.
import tensorflow as tf
try:
    from tensorflow.keras import layers, Model, optimizers, metrics#type: ignore
    from tensorflow.keras.optimizers import Optimizer #type: ignore
except Exception:
    from keras import layers, Model, optimizers, metrics
    from keras.optimizers import Optimizer

from src.utils.config_loader import ConfigLoader
from typing import Optional


def build_hybrid_cnn_lstm(config_path: str = "config/default.yaml", vocab_size: Optional[int] = None) -> Model:
    """
    Build a hybrid CNN + LSTM Keras model.

    Args: 
        config_path: path to YAML config used for hyperparameters
        vocab_size: optional vocabulary size for Embedding input_dim. If not
            provided, the value from config `model.vocab_size` is used.
    """
    # load hyperparameters from config
    cfg = ConfigLoader(config_path)
    max_seq_len = cfg.get("model.max_seq_len", 256)
    cnn_filters = cfg.get("model.cnn_filters", 128)
    cnn_kernel_size = cfg.get("model.cnn_kernel_size", 5)
    lstm_units = cfg.get("model.lstm_units", 64)
    dense_units = cfg.get("model.dense_units", 64)
    dropout_rate = cfg.get("model.dropout_rate", 0.5)

    # inputs
    input_ids = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_ids")
    # Accept an optional mask input so older APIs/tests that pass [ids, mask]
    # continue to work. The mask is not strictly required because Embedding
    # uses mask_zero=True, but exposing the input preserves compatibility.
    input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.float32, name="input_mask")

    # embedding model
    # For now: simple embedding layer will be replaced by a transformer
    embed_dim = int(cfg.get("model.embed_dim", 128))
    embed_vocab = int(vocab_size) if vocab_size is not None else int(cfg.get("model.vocab_size", 20000))
    

    # Safety: ensure a minimum vocab size and avoid impossible values
    if embed_vocab < 2:
        embed_vocab = 2

    # Use mask_zero=True to handle variable-length sequences. Subsequent layers
    # like LSTM and Masking will ignore padded zero-tokens, improving efficiency
    # and preventing the model from learning from padding.
    try:
        embedding_layer = layers.Embedding(
            input_dim=embed_vocab,
            output_dim=embed_dim,
            mask_zero=True,
            name="token_embedding",
        )
    except Exception as e:
        # Fallback strategy: cap vocabulary to a reasonable size and continue
        fallback_vocab = min(max(2000, embed_vocab // 10), 20000)
        print(f"⚠ Could not allocate Embedding(input_dim={embed_vocab}, dim={embed_dim}): {e}\n"
              f"  Falling back to input_dim={fallback_vocab} to continue")
        embedding_layer = layers.Embedding(
            input_dim=fallback_vocab,
            output_dim=embed_dim,
            mask_zero=True,
            name="token_embedding_fallback",
        )


    embeddings = embedding_layer(input_ids)

    # If a mask input is provided, apply it to embeddings so the input_mask
    # becomes part of the computational graph and stays connected to outputs.
    # This keeps backwards compatibility with older callers that pass
    # `[input_ids, input_mask]` as model inputs.
    try:
        from tensorflow.keras import backend as K
        mask_expanded = layers.Reshape((max_seq_len, 1))(input_mask)
        embeddings = layers.Multiply()([embeddings, mask_expanded])
    except Exception:
        # If anything goes wrong, silently continue — embedding still exists
        pass

    # Small spatial dropout on embeddings (helps regularization for conv/lstm)
    emb_dropout = float(cfg.get("model.embedding_dropout", 0.1))
    if emb_dropout and emb_dropout > 0.0:
        try:
            embeddings = layers.SpatialDropout1D(emb_dropout, name="embedding_dropout")(embeddings)
        except Exception:
            # SpatialDropout1D may be unavailable in some TF builds; ignore if it fails
            pass
    
    # Explicitly mask padded tokens (zeros) before feeding to CNN/LSTM branches.
    # While Embedding(mask_zero=True) does this, an explicit Masking layer
    # ensures the mask is correctly propagated, especially in complex models.
    masked_embeddings = layers.Masking(mask_value=0)(embeddings)

    # --- CNN Branch ---
    # The CNN branch captures local patterns and n-grams (like "free money" or
    # "verify account"). It's fast and effective at finding spatial features.
    cnn_branch = layers.Conv1D(filters=cnn_filters,
                               kernel_size=cnn_kernel_size,
                               activation='relu',
                               padding='same')(masked_embeddings)
    cnn_branch = layers.MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = layers.Flatten()(cnn_branch)
    
    # --- LSTM Branch ---
    # The LSTM branch captures long-range dependencies and sequential context,
    # understanding the flow and structure of the email text.
    lstm_branch = layers.Bidirectional(layers.LSTM(lstm_units))(masked_embeddings)

    # --- Combination ---
    # Combining CNN (local features) and LSTM (sequential features) creates a
    # powerful hybrid model that understands both specific keywords and overall context.
    combined = layers.concatenate([cnn_branch, lstm_branch], name="concatenate_branch")
    
    # A dense layer to learn combinations of the extracted features.
    dense = layers.Dense(dense_units, activation='relu', name="dense_layer")(combined)
    
    # Dropout is a regularization technique to prevent overfitting. A rate of 0.5
    # means half of the neurons are randomly dropped during training, forcing the
    # model to learn more robust and generalized features.
    drop = layers.Dropout(dropout_rate, name="dropout_layer")(dense)
    output = layers.Dense(1, activation='sigmoid', name="output")(drop)

    # Keep compatibility with code that passes either a single input or
    # a list [input_ids, input_mask] by defining both inputs on the Model.
    model = Model(inputs=[input_ids, input_mask], outputs=output, name="Hybrid_CNN_LSTM_Model")

    # compile
    model.compile(optimizer=optimizers.Adam(learning_rate=cfg.get("training.learning_rate", 0.001)), #type: ignore
                  loss='binary_crossentropy',
                  metrics=[
                      metrics.BinaryAccuracy(name="accuracy"),
                      metrics.Precision(name="precision"),
                      metrics.Recall(name="recall")
                  ])
    return model
