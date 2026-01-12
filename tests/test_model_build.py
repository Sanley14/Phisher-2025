import numpy as np
from src.model.build_model import build_hybrid_cnn_lstm

def test_build_and_train():
    model = build_hybrid_cnn_lstm("config/default.yaml")
    model.summary()

    # dummy data
    batch_size = 4
    max_seq_len = 256  # ensure this aligns with config
    X_ids = np.random.randint(0, 20000, size=(batch_size, max_seq_len))
    X_mask = np.ones((batch_size, max_seq_len))  # simple mask

    y = np.random.randint(0, 2, size=(batch_size, 1))

    # train for 1 epoch
    model.fit([X_ids, X_mask], y, epochs=1, batch_size=batch_size)
