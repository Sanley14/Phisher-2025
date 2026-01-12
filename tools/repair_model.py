#ensures the model can be loaded and saved in a compatible format from html 5#
import argparse
from pathlib import Path
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import PhishingPredictor




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to existing model file or dir (e.g., models/final_model/model.h5)")
    parser.add_argument("--output", "-o", default="models/final_model_saved", help="Output SavedModel directory")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    print(f"Loading model from: {in_path}")
    try:
        predictor = PhishingPredictor(in_path)
        model = predictor.model
        if model is None:
            print("Loaded predictor but model is None — aborting")
            return

        out_path = str(out_path)
        outp = Path(out_path)
        if outp.exists():
            print(f"Output path {outp} exists; it will be overwritten")

        # If the user requested a .keras or .h5 file, save to that file (Keras format)
        if out_path.endswith('.keras') or out_path.endswith('.h5'):
            print(f"Saving model to Keras file: {outp}")
            try:
                model.save(out_path)
                print(f"Saved model successfully to {out_path}")
            except Exception as e:
                print(f"Failed to save as Keras file: {e}")
        else:
            # Otherwise, save as a SavedModel directory (legacy TF format)
            print(f"Saving model to SavedModel directory: {outp}")
            try:
                model.save(str(outp), save_format="tf")
                print("Saved model successfully.")
            except Exception as e:
                print(f"Failed to save SavedModel directory: {e}")
    except Exception as e:
        print(f"Failed to repair/convert model: {e}")


if __name__ == "__main__":
    main()
