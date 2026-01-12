
import argparse
import h5py
from pathlib import Path
import sys
import traceback


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.build_model import build_hybrid_cnn_lstm


def infer_embedding_input_dim_from_h5(h5_path: str, embedding_layer_names=None):
    """Try to infer embedding input_dim (vocab size) from H5 file by looking for
    embedding weight shapes.
    Returns int or None.
    """
    if embedding_layer_names is None:
        embedding_layer_names = ["simple_embedding", "embedding", "embeddings", "embed"]
    try:
        with h5py.File(h5_path, 'r') as f:
           
            candidates = []
            if 'model_weights' in f:
                root = f['model_weights']
            else:
                root = f
            def walk(group, path=""):
                for key, item in group.items():
                    newpath = f"{path}/{key}" if path else key
                    if isinstance(item, h5py.Group):
                        walk(item, newpath)
                    else:
                      
                        pass
            
            def find_embedding(group):
                for name, item in group.items():
                    if isinstance(item, h5py.Group):
                        
                        if name.lower() in embedding_layer_names:
                            
                            for child_name, child in item.items():
                                try:
                                    shape = child.shape
                                    if isinstance(shape, tuple) and len(shape) >= 2:
                                        
                                        if shape[0] > 1 and shape[1] > 1:
                                            return int(shape[0])
                                except Exception:
                                    pass
                        
                        res = find_embedding(item)
                        if res is not None:
                            return res
                return None
            res = find_embedding(root)
            return res
    except Exception:
        return None


def rebuild(h5_path: str, out_path: str):
    h5 = Path(h5_path)
    out = Path(out_path)

    if not h5.exists():
        print(f"H5 file not found: {h5}")
        return 1

    print(f"Inferring embedding vocab size from: {h5}")
    vocab = infer_embedding_input_dim_from_h5(str(h5))
    print(f"Inferred vocab size: {vocab}")

    print("Building model architecture (using build_hybrid_cnn_lstm)")
    try:
        model = build_hybrid_cnn_lstm(vocab_size=vocab if vocab is not None else None)
    except Exception as e:
        print(f"Failed to build model: {e}")
        traceback.print_exc()
        return 2

    print("Attempting to load weights (by_name=True)...")
    try:
        model.load_weights(str(h5), by_name=True)
        print("Loaded weights by_name=True (partial or full).")
    except Exception as e1:
        print(f"by_name=True failed: {e1}")
        try:
            print("Attempting to load weights (by_name=False)...")
            model.load_weights(str(h5), by_name=False)
            print("Loaded weights by_name=False.")
        except Exception as e2:
            print(f"Failed to load weights from H5: {e2}")
            traceback.print_exc()
            return 3

    
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving rebuilt model to: {out}")
        model.save(str(out))
        print("Saved rebuilt model successfully.")
        return 0
    except Exception as e:
        print(f"Failed to save rebuilt model: {e}")
        traceback.print_exc()
        return 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True, help='Path to H5 model file (full-model .h5)')
    parser.add_argument('--out', required=True, help='Output .keras path for rebuilt model')
    args = parser.parse_args()

    rc = rebuild(args.h5, args.out)
    sys.exit(rc)
