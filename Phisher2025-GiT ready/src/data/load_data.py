
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class PhishingDataLoader:
       
    def __init__(self, max_seq_len: int = 256, vocab_size: int = 20000):
        """
        Initialize data loader.
        
        Args:
            max_seq_len: Maximum sequence length for padding/truncation
            vocab_size: Size of vocabulary for tokenization
        """
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        
        self.tokenizer = None
        self._init_tokenizer()
    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        try:
            
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            
            print("✓ Loaded XLM-RoBERTa tokenizer")
        except Exception as e:
            print(f"⚠ Could not load XLM-RoBERTa: {e}")
            print("  Using simple whitespace tokenizer as fallback")
            self.tokenizer = None
    
    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: email_text, language, label
            
        Returns:
            DataFrame with dataset
        """
        print(f"Loading dataset from {csv_path}...")
        
        
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
        
        return df
    
    def tokenize_text(self, text: str) -> np.ndarray:
        """
        Tokenize a single text.
        
        Args:
            text: Email text to tokenize
            
        Returns:
            Array of token IDs
        """
        if self.tokenizer is not None:
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            ids = np.array(encoded["input_ids"][0], dtype=np.int64)
            
            if ids.max() >= self.vocab_size:
                ids = ids % self.vocab_size
            return ids
        else:
            
            tokens = text.split()[:self.max_seq_len]
            
            
            token_ids = [hash(t) % self.vocab_size for t in tokens]
            
          
            token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
            
            return np.array(token_ids[:self.max_seq_len])
    
    def prepare_texts(self, texts: List[str]) -> np.ndarray:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of email texts
            
        Returns:
            2D array of shape (len(texts), max_seq_len)
        """
        print(f"Tokenizing {len(texts)} texts...")
        
        
        tokenized = np.array([self.tokenize_text(text) for text in texts])
        
        print(f"✓ Tokenized shape: {tokenized.shape}")
        
        return tokenized
    
    def split_dataset(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            df: Full dataset
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        
        # Ensure proportions sum to 1.0
        total = train_size + val_size + test_size
        train_size /= total
        val_size /= total
        test_size /= total
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=df["label"]  # Ensure balanced split
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
            stratify=temp_df["label"]
        )
        
        print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_tf_dataset(
        self,
        texts: List[str],
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for training/evaluation.
        
        Args:
            texts: List of email texts
            labels: Array of labels (0 or 1)
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset
        """
        # Tokenize texts
        X = self.prepare_texts(texts)
        y = labels.astype(np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(texts))
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"✓ Created TF dataset with batch_size={batch_size}")
        
        return dataset
    
    def prepare_full_pipeline(
        self,
        csv_path: str,
        batch_size: int = 32,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Complete pipeline: load data, split, tokenize, create batches.
        
        Args:
            csv_path: Path to CSV dataset
            batch_size: Batch size for datasets
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("\n" + "=" * 70)
        print("FULL DATA PIPELINE")
        print("=" * 70)
        
        # Load
        df = self.load_dataset(csv_path)
        
        # Split
        train_df, val_df, test_df = self.split_dataset(
            df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size
        )
        
        # Create datasets
        train_dataset = self.create_tf_dataset(
            train_df["email_text"].tolist(),
            train_df["label"].values,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataset = self.create_tf_dataset(
            val_df["email_text"].tolist(),
            val_df["label"].values,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_dataset = self.create_tf_dataset(
            test_df["email_text"].tolist(),
            test_df["label"].values,
            batch_size=batch_size,
            shuffle=False
        )
        
        print("\n✓ Data pipeline complete!")
        
        # Save split info
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return train_dataset, val_dataset, test_dataset
