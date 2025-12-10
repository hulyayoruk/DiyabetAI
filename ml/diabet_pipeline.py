import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ==========================
# 1. VERİYİ YÜKLE & TARİHLERİ PARSE ET
# ==========================

def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Tarih kolonlarını datetime'e çevir (dayfirst=True)
    df["ts_dt"]       = pd.to_datetime(df["ts"],       dayfirst=True, errors="coerce")
    df["ts_begin_dt"] = pd.to_datetime(df["ts_begin"], dayfirst=True, errors="coerce")
    df["ts_end_dt"]   = pd.to_datetime(df["ts_end"],   dayfirst=True, errors="coerce")
    df["tbegin_dt"]   = pd.to_datetime(df["tbegin"],   dayfirst=True, errors="coerce")
    df["tend_dt"]     = pd.to_datetime(df["tend"],     dayfirst=True, errors="coerce")

    # Genel event zamanı: önce ts, yoksa ts_begin, yoksa tbegin
    df["event_time"] = df["ts_dt"]
    mask = df["event_time"].isna() & df["ts_begin_dt"].notna()
    df.loc[mask, "event_time"] = df.loc[mask, "ts_begin_dt"]
    mask = df["event_time"].isna() & df["tbegin_dt"].notna()
    df.loc[mask, "event_time"] = df.loc[mask, "tbegin_dt"]

    return df


# ==========================
# 2. TEK HASTA İÇİN ZAMAN SERİSİ ÇERÇEVESİ
# ==========================

def build_patient_timeseries(df_all: pd.DataFrame,
                             patient_id: int,
                             freq: str = "5min",
                             horizon_steps: int = 6) -> pd.DataFrame:
    """
    freq  : 5min = 5 dakikalık grid
    horizon_steps : 6 adım sonrası (6*5dk = 30dk) glikoz tahmini
    """
    df_p = df_all[df_all["patient_id"] == patient_id].copy()

    # --- ANA GLİKOZ ZAMAN SERİSİ ---
    g = df_p[df_p["category"] == "glucose_level"].copy()
    g = g.dropna(subset=["ts_dt"])
    g = g.sort_values("ts_dt")
    g["glucose"] = pd.to_numeric(g["value"], errors="coerce")

    # 5 dakikalık grid -> eksik zamanları doldur
    g = g.set_index("ts_dt").resample(freq).asfreq()
    g["glucose"] = g["glucose"].interpolate(limit_direction="both")

    # Hasta bilgileri
    g["patient_id"] = patient_id
    g["weight"] = df_p["weight"].iloc[0] if not df_p["weight"].isna().all() else np.nan

    # --- MEAL (ÖĞÜN) ---
    meal = df_p[df_p["category"] == "meal"].copy()
    if not meal.empty:
        meal = meal.dropna(subset=["event_time"])
        meal["carbs"] = pd.to_numeric(meal["carbs"], errors="coerce")
        meal = meal.set_index("event_time").sort_index()
        g["carbs_event"] = meal["carbs"]
    else:
        g["carbs_event"] = np.nan

    g["carbs_1h"] = g["carbs_event"].rolling("60min").sum()
    g["carbs_2h"] = g["carbs_event"].rolling("120min").sum()
    g["carbs_4h"] = g["carbs_event"].rolling("240min").sum()

    # --- BOLUS İNSÜLİN ---
    bolus = df_p[df_p["category"] == "bolus"].copy()
    if not bolus.empty:
        bolus = bolus.dropna(subset=["event_time"])
        # Veri setinde bwz_carb_input kolonunda dozlar var
        bolus["bolus_dose"] = pd.to_numeric(bolus["bwz_carb_input"], errors="coerce")
        bolus = bolus.set_index("event_time").sort_index()
        g["bolus_event"] = bolus["bolus_dose"]
    else:
        g["bolus_event"] = np.nan

    g["bolus_2h"] = g["bolus_event"].rolling("120min").sum()
    g["bolus_4h"] = g["bolus_event"].rolling("240min").sum()

    # --- BASAL İNSÜLİN ---
    basal = df_p[df_p["category"] == "basal"].copy()
    if not basal.empty:
        basal = basal.dropna(subset=["event_time"])
        basal["basal_rate"] = pd.to_numeric(basal["value"], errors="coerce")
        basal = basal.set_index("event_time").sort_index()
        g["basal_rate"] = basal["basal_rate"].reindex(g.index).ffill()
    else:
        g["basal_rate"] = np.nan

    # TEMP BASAL (varsa basal_rate'i override edebilir)
    temp = df_p[df_p["category"] == "temp_basal"].copy()
    if not temp.empty:
        temp["temp_val"] = pd.to_numeric(temp["value"], errors="coerce")
        g["temp_basal_active"] = 0.0
        for _, row in temp.dropna(subset=["ts_begin_dt", "ts_end_dt"]).iterrows():
            mask = (g.index >= row["ts_begin_dt"]) & (g.index <= row["ts_end_dt"])
            g.loc[mask, "temp_basal_active"] = 1.0
        # basal_rate aktif değilken 0 olsun
        g["basal_effective"] = np.where(g["temp_basal_active"] == 1.0,
                                        0.0,
                                        g["basal_rate"])
    else:
        g["temp_basal_active"] = 0.0
        g["basal_effective"] = g["basal_rate"]

    # --- EGZERSİZ ---
    ex = df_p[df_p["category"] == "exercise"].copy()
    if not ex.empty:
        ex = ex.dropna(subset=["event_time"])
        ex["ex_duration"] = pd.to_numeric(ex["value"], errors="coerce")
        ex["ex_intensity"] = pd.to_numeric(ex["intensity"], errors="coerce")
        ex = ex.set_index("event_time").sort_index()
        g["ex_duration_event"] = ex["ex_duration"]
        g["ex_intensity_event"] = ex["ex_intensity"]
    else:
        g["ex_duration_event"] = np.nan
        g["ex_intensity_event"] = np.nan

    g["exercise_1h"] = g["ex_duration_event"].rolling("60min").sum()
    g["exercise_intensity_mean_1h"] = g["ex_intensity_event"].rolling("60min").mean()

    # --- BASIS STEPS ---
    steps = df_p[df_p["category"] == "basis_steps"].copy()
    if not steps.empty:
        steps = steps.dropna(subset=["event_time"])
        steps["steps_val"] = pd.to_numeric(steps["value"], errors="coerce")
        steps = steps.set_index("event_time").sort_index()
        g["steps"] = steps["steps_val"].reindex(g.index).fillna(0)
    else:
        g["steps"] = 0.0

    g["steps_30min"] = g["steps"].rolling("30min").sum()

    # --- UYKU (basis_sleep + sleep) ---
    g["is_sleep"] = 0

    bs = df_p[df_p["category"] == "basis_sleep"].copy()
    if not bs.empty:
        for _, row in bs.dropna(subset=["tbegin_dt", "tend_dt"]).iterrows():
            mask = (g.index >= row["tbegin_dt"]) & (g.index <= row["tend_dt"])
            g.loc[mask, "is_sleep"] = 1

    sl = df_p[df_p["category"] == "sleep"].copy()
    if not sl.empty:
        for _, row in sl.dropna(subset=["ts_begin_dt", "ts_end_dt"]).iterrows():
            mask = (g.index >= row["ts_begin_dt"]) & (g.index <= row["ts_end_dt"])
            g.loc[mask, "is_sleep"] = 1

    # --- ZAMAN FEATURES ---
    g["minute_of_day"] = g.index.hour * 60 + g.index.minute
    g["time_sin"] = np.sin(2 * np.pi * g["minute_of_day"] / (24 * 60))
    g["time_cos"] = np.cos(2 * np.pi * g["minute_of_day"] / (24 * 60))

    # --- HEDEF: 30 dakika sonrası glikoz ---
    g["glucose_future"] = g["glucose"].shift(-horizon_steps)

    # BASİT RİSK ETİKETLERİ
    g["hypo_risk_30m"] = (g["glucose_future"] < 70).astype(float)
    g["hyper_risk_30m"] = (g["glucose_future"] > 180).astype(float)

    conds = [
        g["glucose_future"] < 70,
        (g["glucose_future"] >= 70) & (g["glucose_future"] <= 180),
        g["glucose_future"] > 180,
    ]
    choices = [0, 1, 2]
    g["risk_class_30m"] = np.select(conds, choices, default=np.nan)

    return g


# ==========================
# 3. LSTM DATASET HAZIRLAMA
# ==========================

def make_lstm_dataset(g: pd.DataFrame, seq_len: int = 12):
    """
    seq_len: geçmiş kaç adım (12*5dk = 60dk)
    """
    feature_cols = [
        "glucose",
        "carbs_1h", "carbs_2h", "carbs_4h",
        "bolus_2h", "bolus_4h",
        "basal_effective",
        "exercise_1h", "exercise_intensity_mean_1h",
        "steps_30min",
        "is_sleep",
        "time_sin", "time_cos",
    ]

    # Sadece glikoz ve hedefin boş olmamasını zorunlu yap
    g_model = g.dropna(subset=["glucose", "glucose_future"]).copy()

    # Diğer feature'larda NaN varsa 0 ile doldur
    for col in feature_cols:
        if col != "glucose":
            g_model[col] = g_model[col].fillna(0)

    data = g_model[feature_cols].values
    target_reg = g_model["glucose_future"].values
    target_hypo = g_model["hypo_risk_30m"].fillna(0).values

    X_seq, y_reg, y_hypo = [], [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i+seq_len, :])
        y_reg.append(target_reg[i+seq_len])
        y_hypo.append(target_hypo[i+seq_len])

    return np.array(X_seq), np.array(y_reg), np.array(y_hypo)


# ==========================
# 4. XGBOOST DATASET HAZIRLAMA
# ==========================

def add_tabular_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    g["glucose_lag1"] = g["glucose"].shift(1)
    g["glucose_lag2"] = g["glucose"].shift(2)
    g["glucose_lag6"] = g["glucose"].shift(6)  # 30 dk önce

    g["glucose_mean_1h"] = g["glucose"].rolling("60min").mean()
    g["glucose_min_1h"]  = g["glucose"].rolling("60min").min()
    g["glucose_max_1h"]  = g["glucose"].rolling("60min").max()

    g["glucose_diff_30m"] = g["glucose"] - g["glucose_lag6"]

    return g


def make_xgb_dataset(g: pd.DataFrame):
    g = add_tabular_features(g)

    feature_cols = [
        "glucose",
        "glucose_lag1", "glucose_lag2", "glucose_lag6",
        "glucose_mean_1h", "glucose_min_1h", "glucose_max_1h",
        "glucose_diff_30m",
        "carbs_1h", "carbs_2h", "carbs_4h",
        "bolus_2h", "bolus_4h",
        "basal_effective",
        "exercise_1h", "exercise_intensity_mean_1h",
        "steps_30min",
        "is_sleep",
        "time_sin", "time_cos",
    ]

    # Yine glikoz + risk sınıfını zorunlu yapıyoruz
    g_model = g.dropna(subset=["glucose", "risk_class_30m"]).copy()

    for col in feature_cols:
        if col != "glucose":
            g_model[col] = g_model[col].fillna(0)

    X = g_model[feature_cols].values
    y = g_model["risk_class_30m"].astype(int).values

    return X, y


# ==========================
# 5. TÜM HASTALAR İÇİN DATASET OLUŞTUR
# ==========================

def build_datasets_for_all_patients(csv_path: str,
                                    seq_len: int = 12,
                                    horizon_steps: int = 6):
    df_all = load_events(csv_path)

    X_lstm_list, y_lstm_list = [], []
    X_xgb_list, y_xgb_list = [], []

    for pid in df_all["patient_id"].dropna().unique():
        print(f"Preparing data for patient {pid}...")
        g = build_patient_timeseries(df_all, pid, freq="5min", horizon_steps=horizon_steps)

        # LSTM
        X_seq, y_reg, _ = make_lstm_dataset(g, seq_len=seq_len)
        print(f"Patient {pid} - LSTM samples: {len(X_seq)}")
        if len(X_seq) > 0:
            X_lstm_list.append(X_seq)
            y_lstm_list.append(y_reg)

        # XGBoost
        X_tab, y_tab = make_xgb_dataset(g)
        print(f"Patient {pid} - XGB samples: {len(X_tab)}")
        if len(X_tab) > 0:
            X_xgb_list.append(X_tab)
            y_xgb_list.append(y_tab)

    if not X_lstm_list or not X_xgb_list:
        raise RuntimeError("Hiç sample üretilmedi, feature/NaN mantığını tekrar kontrol et.")

    X_lstm = np.concatenate(X_lstm_list, axis=0)
    y_lstm = np.concatenate(y_lstm_list, axis=0)
    X_xgb  = np.concatenate(X_xgb_list, axis=0)
    y_xgb  = np.concatenate(y_xgb_list, axis=0)

    return X_lstm, y_lstm, X_xgb, y_xgb


# ==========================
# 6. LSTM MODEL EĞİTİMİ
# ==========================

def train_lstm(X, y, epochs: int = 20, batch_size: int = 64):
    num_samples, seq_len, num_features = X.shape

    scaler = StandardScaler()
    X_flat = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(num_samples, seq_len, num_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = Sequential([
        LSTM(64, input_shape=(seq_len, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer=Adam(1e-3))
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    y_pred = model.predict(X_test).flatten()
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"LSTM Test MAE: {mae:.2f} mg/dL")

    return model, scaler


# ==========================
# 7. XGBOOST MODEL EĞİTİMİ
# ==========================

def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )

    xgb_clf.fit(X_train, y_train)

    y_pred = xgb_clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"XGBoost accuracy: {acc:.3f}")

    return xgb_clf


# ==========================
# 8. ÖRNEK ÇALIŞTIRMA
# ==========================

if __name__ == "__main__":
    # İstersen burayı kendi absolute path’inle bırakabilirsin:
    csv_path = r"C:\Users\hulya\OneDrive\Masaüstü\Diyabet-AI\mühendislik projesi\data\ohio\all_patients_events.csv"

    print("Building datasets...")
    X_lstm, y_lstm, X_xgb, y_xgb = build_datasets_for_all_patients(csv_path)

    print("Training LSTM...")
    lstm_model, lstm_scaler = train_lstm(X_lstm, y_lstm)

    print("Training XGBoost...")
    xgb_model = train_xgb(X_xgb, y_xgb)
 # ==== MODELLERİ KAYDET ====
    import os
    import joblib

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # LSTM modeli (.h5)
    lstm_path = os.path.join(models_dir, "lstm_glucose_30min.h5")
    lstm_model.save(lstm_path)
    print("LSTM model saved to:", lstm_path)

    # LSTM scaler (pickle/joblib)
    scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")
    joblib.dump(lstm_scaler, scaler_path)
    print("LSTM scaler saved to:", scaler_path)

    # XGBoost modeli (.json)
    xgb_path = os.path.join(models_dir, "xgb_risk_30min.json")
    xgb_model.save_model(xgb_path)
    print("XGBoost model saved to:", xgb_path)

    print("\n*** UYARI ***")
    print("Bu modeller SADECE veri tabanlı risk/tahmin amaçlıdır, tıbbi tanı veya tedavi önerisi DEĞİLDİR.")