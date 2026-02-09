import os

import joblib
import pandas as pd


def load_model(input_dir):
    for name in os.listdir(input_dir):
        if name.endswith((".pkl", ".joblib")):
            return joblib.load(os.path.join(input_dir, name))
    raise FileNotFoundError(f"No model file in {input_dir}")


def predict(context, data):
    if hasattr(context, "predict_proba"):
        preds = context.predict_proba(data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data)
        return pd.DataFrame(preds, columns=getattr(context, "classes_", ["class_0", "class_1"]))
    preds = context.predict(data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data)
    return pd.DataFrame(preds, columns=["predictions"])
