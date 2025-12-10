# ml/train_xgb_risk.py

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .feature_config import XGB_FEATURE_ORDER

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_training_data():
    """
    Burada kendi dataset'ini okuyacaksın.
    Örneğin:
      - daha önce oluşturduğun bir CSV
      - ya da SQL'den export edilmiş bir tablo

    Beklenen kolonlar:
      glucose, carbs_win_1, carbs_win_2, carbs_win_3,
      bolus, bolus_corr, basal,
      ex_minutes, ex_intensity, steps,
      is_sleep, time_sin, time_cos,
      risk_label  (0=hypo, 1=normal, 2=hyper)

    Aşağıda örnek bir CSV path'i var; kendine göre değiştir.
    """
    data_path = os.path.join(BASE_DIR, "data", "risk_training_data.csv")
    df = pd.read_csv(data_path)

    # Feature'lar
    X = df[XGB_FEATURE_ORDER].astype(float).values

    # Hedef / label (0,1,2)
    y = df["risk_label"].astype(int).values

    return X, y


def train_and_save():
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    # Basit rapor
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Modeli kaydet
    model_path = os.path.join(MODELS_DIR, "xgb_risk_30min.json")
    model.save_model(model_path)
    print(f"XGB model kaydedildi: {model_path}")

    # Feature sırasını da JSON olarak kaydedelim (doküman gibi)
    feature_order_path = os.path.join(
        MODELS_DIR, "xgb_feature_order.json"
    )
    with open(feature_order_path, "w", encoding="utf-8") as f:
        json.dump(XGB_FEATURE_ORDER, f, ensure_ascii=False, indent=2)
    print(f"Feature sırası kaydedildi: {feature_order_path}")


if __name__ == "__main__":
    train_and_save()
