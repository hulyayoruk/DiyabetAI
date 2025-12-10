# ml/serve_models.py

import os
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier

# ==========================
#  MODEL YÜKLEME YOLLARI
# ==========================
# BASE_DIR  -> proje kökü (Diyabet-AI)
# MODELS_DIR -> Diyabet-AI/models
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Yolları bir kere ekrana da yazalım (debug için istersen)
print("MODELS_DIR:", MODELS_DIR)

# --- LSTM ---
lstm_path = os.path.join(MODELS_DIR, "lstm_glucose_30min.h5")
scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")

if not os.path.exists(lstm_path):
    raise FileNotFoundError(f"LSTM modeli bulunamadı: {lstm_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"LSTM scaler bulunamadı: {scaler_path}")

lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
lstm_scaler = joblib.load(scaler_path)

# Eğitim sırasında gördüğün MAE (istersen dosyadan da okuyabilirsin)
lstm_mae = 4.97

# --- XGBoost ---
xgb_path = os.path.join(MODELS_DIR, "xgb_risk_30min.json")
if not os.path.exists(xgb_path):
    raise FileNotFoundError(f"XGB model dosyası bulunamadı: {xgb_path}")

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_path)


# ==========================
#  YARDIMCI FONKSİYON
# ==========================
def _match_feature_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Gelen feature sayısını modele uydur (gerekirse pad / kes)."""
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] < target_dim:
        pad = np.zeros((X.shape[0], target_dim - X.shape[1]))
        return np.hstack([X, pad])
    return X[:, :target_dim]


# ==========================
#  ✔ LSTM REGRESYON
# ==========================
def predict_lstm(window_12xF: np.ndarray) -> float:
    """
    window_12xF: (12, n_features) zaman serisi penceresi
    """
    seq_len, n_feat = window_12xF.shape

    scaler_dim = lstm_scaler.mean_.shape[0]
    flat = window_12xF.reshape(-1, n_feat)
    flat = _match_feature_dim(flat, scaler_dim)

    scaled = lstm_scaler.transform(flat).reshape(1, seq_len, scaler_dim)
    pred = lstm_model.predict(scaled).flatten()[0]
    return float(pred)


# ==========================
#  ✔ XGB RISK OLASILIK
# ==========================
def predict_xgb(x: np.ndarray) -> np.ndarray:
    """
    x: (n_samples, n_features) – XGBoost eğitiminde kullandığın feature vektörü.
    """
    xgb_dim = xgb_model.n_features_in_
    x = _match_feature_dim(x, xgb_dim)
    probs = xgb_model.predict_proba(x)
    return probs


# ==========================
#  1 Fonksiyon — 6 Bilgi
# ==========================
def predict_all(window_12xF: np.ndarray, xgb_features: np.ndarray | None = None):
    """
    window_12xF : LSTM için (12, n_features) pencere
    xgb_features: XGBoost için (n_features,) veya (1, n_features) vektör.
                  Bunu app.py içindeki build_xgb_features üretebiliyor.
    """
    # 1️⃣ LSTM tahmini
    pred_glucose = predict_lstm(window_12xF)

    # 2️⃣ Tahmin aralığı
    lower = pred_glucose - lstm_mae
    upper = pred_glucose + lstm_mae

    # 3️⃣ Trend için mevcut glikoz
    current = window_12xF[-1, 0]
    delta = pred_glucose - current
    if delta > 5:
        trend = "rising"
    elif delta < -5:
        trend = "falling"
    else:
        trend = "stable"

    # 4️⃣ Risk olasılıkları (XGBoost)
    if xgb_features is not None:
        if xgb_features.ndim == 1:
            xgb_in = xgb_features.reshape(1, -1)
        else:
            xgb_in = xgb_features
        probs = predict_xgb(xgb_in)[0]
    else:
        # Geriye dönük uyumluluk için; ama normalde buraya düşmemeliyiz
        probs = predict_xgb(window_12xF[-1:, :])[0]

    p_hypo, p_normal, p_hyper = [float(v) for v in probs.tolist()]

    # 5️⃣ Risk sınıfı (en yüksek olasılık)
    risk_class = int(np.argmax(probs))

    # Model tabanlı risk skoru
    risk_score = float(p_hyper * 1.0 + p_hypo * 0.35)

    return {
        "prediction": float(pred_glucose),
        "lower": float(lower),
        "upper": float(upper),
        "trend": trend,
        "delta": float(delta),
        "risk_class": risk_class,
        "p_hypo": p_hypo,
        "p_normal": p_normal,
        "p_hyper": p_hyper,
        "risk_score": risk_score,
    }
