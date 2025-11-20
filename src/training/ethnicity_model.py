# src/training/ethnicity_model.py
"""
ethnicity_model.py
-------------------

Defines storage + utilities for the LightGBM ethnicity classifier.

Responsibilities:
- Provide method to train a LightGBM model
- Save/load:
    - LightGBM booster
    - LabelEncoder
- Prepare evaluation metrics

Tools:
- lightgbm
- scikit-learn

TODO:
- Parameter tuning grid
- Add confusion matrix visualization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import joblib


def train_ethnicity(X, y, model_path, encoder_path):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    params = {"objective": "multiclass", "num_class": len(le.classes_)}

    train_data = lgb.Dataset(X, label=y_enc)
    model = lgb.train(params, train_data, num_boost_round=200)

    model.save_model(model_path)
    joblib.dump(le, encoder_path)
