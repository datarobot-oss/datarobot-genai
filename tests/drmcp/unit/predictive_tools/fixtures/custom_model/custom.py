# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
