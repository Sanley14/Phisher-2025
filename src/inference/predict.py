"""
Inference module for making predictions on new emails.

Loads a trained model and provides functions to:
1. Predict single email
2. Batch predict multiple emails  
3. Get confidence scores
4. Explain predictions (language detection)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
from transformers import AutoTokenizer

#Make predictions using trained phishing detection model#
class PhishingPredictor:
    
    def __init__(self, model_path: str):
        """Initialize predictor with trained model.

        Args:
            model_path: Path to trained model file or directory (supports
                native Keras `.keras`, H5 `.h5`, or SavedModel dir).
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.embedding_vocab_size = 20000  # default fallback

        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        # After loading model, try to infer embedding vocab size so we can
        # map tokenizer ids into the model embedding range during inference.
        try:
            for layer in self.model.layers:
                # Keras Embedding layers expose `input_dim`
                if hasattr(layer, "input_dim") and getattr(layer, "input_dim") is not None:
                    self.embedding_vocab_size = int(layer.input_dim)
                    break
        except Exception:
            pass
    
    def _load_model(self):
        """Load trained model from disk."""
        
        print(f"Loading model from {self.model_path}...")
        # Try different model formats
        try:
            # Try loading directly (works for .keras, .h5, or SavedModel dirs)
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("✓ Loaded model from given path")
            return
        except Exception as e1:
            err_msg = str(e1)
            print(f"Initial load failed: {err_msg[:200]}")

            # Common H5 issue: unknown serialized layer names (e.g., 'NotEqual')
            # Provide a small compatibility shim and retry with custom_objects.
            def _retry_with_custom_objects(path):
                # Define compatibility layers for some TF op-turned-layer names
                class NotEqualLayer(tf.keras.layers.Layer):
                    def __init__(self, **kwargs):
                        # Accept arbitrary kwargs from deserialization, but avoid
                        # passing unknown args to base Layer.__init__ which will
                        # raise for unexpected keywords.
                        name = kwargs.pop("name", None)
                        dtype = kwargs.pop("dtype", None)
                        trainable = kwargs.pop("trainable", None)
                        if name is not None or dtype is not None or trainable is not None:
                            # Only pass known args
                            init_kwargs = {}
                            if name is not None:
                                init_kwargs["name"] = name
                            if dtype is not None:
                                init_kwargs["dtype"] = dtype
                            if trainable is not None:
                                init_kwargs["trainable"] = trainable
                            super().__init__(**init_kwargs)
                        else:
                            super().__init__()
                        # store remaining config so get_config can return it
                        self._saved_config = kwargs.copy()
                        if name is not None:
                            self._saved_config["name"] = name

                    def call(self, inputs):
                        try:
                            if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                                return tf.math.not_equal(inputs[0], inputs[1])
                            return tf.math.not_equal(inputs, 0)
                        except Exception:
                            inp = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                            # Fallback: return zeros of appropriate dtype
                            try:
                                return tf.zeros_like(inp)
                            except Exception:
                                return tf.zeros([1])

                    def get_config(self):
                        # Return saved config so deserializer can reconstruct kwargs
                        return dict(self._saved_config if hasattr(self, "_saved_config") else {})

                class PassThroughLayer(tf.keras.layers.Layer):
                    def __init__(self, **kwargs):
                        # Accept arbitrary kwargs from deserialization; pull out
                        # known Layer args and store the rest for get_config.
                        name = kwargs.pop("name", None)
                        dtype = kwargs.pop("dtype", None)
                        trainable = kwargs.pop("trainable", None)
                        if name is not None or dtype is not None or trainable is not None:
                            init_kwargs = {}
                            if name is not None:
                                init_kwargs["name"] = name
                            if dtype is not None:
                                init_kwargs["dtype"] = dtype
                            if trainable is not None:
                                init_kwargs["trainable"] = trainable
                            super().__init__(**init_kwargs)
                        else:
                            super().__init__()
                        # store remaining config
                        self._saved_config = kwargs.copy()
                        if name is not None:
                            self._saved_config["name"] = name

                    def call(self, inputs):
                        # Return inputs unchanged (or first element if list)
                        if isinstance(inputs, (list, tuple)):
                            return inputs[0]
                        return inputs

                    def get_config(self):
                        return dict(self._saved_config if hasattr(self, "_saved_config") else {})

                # Common names we've seen serialized in problematic H5 exports
                candidate_names = [
                    "NotEqual", "not_equal",
                    "Any", "AnyLayer", "AnyOp",
                    "Any_0", "Any_1", "AnyBoolean",
                ]

                custom_objects = {"NotEqual": NotEqualLayer, "not_equal": tf.math.not_equal}
                for n in candidate_names:
                    if n not in custom_objects:
                        custom_objects[n] = PassThroughLayer
                try:
                    m = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                    print(f"✓ Loaded model from {path} using compatibility custom_objects")
                    return m
                except Exception as e:
                    raise e

            # If direct load failed, try common locations inside a model folder
            p = Path(self.model_path)
            tried = []
            if p.is_dir():
                candidates = [p / "model.keras", p / "model.h5", p / "model_saved", p]
            else:
                candidates = [p, p.parent / "model.keras", p.parent / "model.h5", p.parent / "model_saved"]

            for c in candidates:
                try:
                    cstr = str(c)
                    # First try without custom_objects (some paths may work)
                    try:
                        self.model = tf.keras.models.load_model(cstr, compile=False)
                        print(f"✓ Loaded model from {cstr}")
                        return
                    except Exception:
                        # Retry with custom_objects compatibility shim
                        self.model = _retry_with_custom_objects(cstr)
                        return
                except Exception as e:
                    tried.append((c, str(e)))

            msgs = " ; ".join([f"{c}: {err[:200]}" for c, err in tried])
            raise RuntimeError(f"Could not load model. Attempts: {msgs}")
    
    def _load_tokenizer(self):
        """Load tokenizer (should match training tokenizer)."""
        
        try:
            # Use XLM-RoBERTa (same as training)
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            print("✓ Loaded XLM-RoBERTa tokenizer")
        except Exception as e:
            print(f"⚠ Could not load XLM-RoBERTa: {e}")
            print("  Will use fallback tokenization")
            self.tokenizer = None
    
    def tokenize_text(self, text: str, max_seq_len: int = 256) -> np.ndarray:
        """
        Tokenize a single email text.
        
        Args:
            text: Email text to tokenize
            max_seq_len: Maximum sequence length
            
        Returns:
            Array of token IDs
        """
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            ids = np.array(encoded["input_ids"][0], dtype=np.int64)
            # If ids exceed the model embedding vocab size, map them into the
            # embedding range to avoid out-of-bounds errors (best-effort).
            if ids.max() >= self.embedding_vocab_size:
                ids = ids % max(1, self.embedding_vocab_size)
            return ids
        else:
            # Fallback: simple whitespace tokenization
            tokens = text.split()[:max_seq_len]
            token_ids = [hash(t) % self.embedding_vocab_size for t in tokens]
            token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
            return np.array(token_ids[:max_seq_len])
    
    def predict_single(self, text: str) -> Tuple[float, str]:
        """
        Predict phishing probability for a single email.
        
        Args:
            text: Email text
            
        Returns:
            Tuple of (confidence_score, label)
            - confidence_score: Float between 0-1 (higher = more phishing)
            - label: 'PHISHING' or 'LEGITIMATE'
        """
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Add batch dimension
        tokens = np.expand_dims(tokens, axis=0)
        
        # Predict
        prediction = self.model.predict(tokens, verbose=0)
        
        # Extract score (should be between 0-1 after sigmoid)
        score = float(prediction[0][0])
        
        # Classify
        label = "PHISHING" if score > 0.5 else "LEGITIMATE"
        
        return score, label
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[float, str]]:
        """
        Predict phishing probability for multiple emails.
        
        Args:
            texts: List of email texts
            
        Returns:
            List of (confidence_score, label) tuples
        """
        print(f"Predicting {len(texts)} emails...")
        
        # Tokenize all texts
        tokens_list = []
        for text in texts:
            tokens = self.tokenize_text(text)
            tokens_list.append(tokens)
        
        # Convert to array
        tokens_array = np.array(tokens_list)
        
        # Batch predict
        predictions = self.model.predict(tokens_array, verbose=0)
        
        # Process results
        results = []
        for score in predictions:
            score_val = float(score[0])
            label = "PHISHING" if score_val > 0.5 else "LEGITIMATE"
            results.append((score_val, label))
        
        print(f"✓ Predictions complete")
        
        return results
    
    def predict_with_confidence(self, text: str, threshold: float = 0.5) -> dict:
        """
        Predict with detailed confidence information.
        
        Args:
            text: Email text
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with prediction details
        """
        score, label = self.predict_single(text)
        
        # Calculate confidence
        if score > threshold:
            confidence = score
        else:
            confidence = 1.0 - score
        
        return {
            "raw_score": score,
            "label": label,
            "confidence": confidence,
            "threshold": threshold,
            "is_phishing": label == "PHISHING"
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def predict_email(text: str, model_path: str = "models/final_model/model.h5") -> dict:
    """
    Simple one-liner to predict a single email.
    
    Args:
        text: Email text
        model_path: Path to model
        
    Returns:
        Prediction result dictionary
    """
    predictor = PhishingPredictor(model_path)
    return predictor.predict_with_confidence(text)


def batch_predict_emails(
    texts: List[str],
    model_path: str = "models/final_model/model.h5"
) -> List[dict]:
    """
    Simple one-liner to predict multiple emails.
    
    Args:
        texts: List of email texts
        model_path: Path to model
        
    Returns:
        List of prediction result dictionaries
    """
    predictor = PhishingPredictor(model_path)
    results = predictor.predict_batch(texts)
    
    # Convert to list of dicts
    return [
        {
            "raw_score": score,
            "label": label,
            "is_phishing": label == "PHISHING"
        }
        for score, label in results
    ]


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================


def main():
    """Test the predictor on sample emails."""
    
    print("=" * 70)
    print("PHISHING DETECTOR - INFERENCE TEST")
    print("=" * 70)
    
    # Sample emails
    test_emails = [
        "Urgent: Verify your PayPal account immediately. Click here: bit.ly/verify2024",
        "Thank you for using our service. Here is your monthly invoice.",
        "WARNING: Your bank account will be suspended unless you confirm here: bit.ly/confirm",
        "We appreciate your business. See your order status below.",
    ]
    
    # Try to predict
    try:
        model_path = "models/final_model/model.h5"
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"\n⚠ Model not found at {model_path}")
            print("  Please train the model first using: python src/model/train_baseline.py")
            return
        
        # Create predictor
        predictor = PhishingPredictor(model_path)
        
        # Test predictions
        print("\nTest Predictions:")
        print("-" * 70)
        
        for i, email in enumerate(test_emails, 1):
            result = predictor.predict_with_confidence(email)
            
            print(f"\n[Email {i}]")
            print(f"Text: {email[:60]}...")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Raw Score: {result['raw_score']:.4f}")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
