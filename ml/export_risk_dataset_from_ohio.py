# ml/export_risk_dataset_from_ohio.py

"""
Ohio verisinden, XGBoost risk modeli için eğitim dataseti üretir.

Çıktı:
  <proje_kök>/data/risk_training_data.csv

Kolonlar:
  glucose, carbs_win_1, carbs_win_2, carbs_win_3,
  bolus, bolus_corr, basal,
  ex_minutes, ex_intensity, steps,
  is_sleep, time_sin, time_cos,
  risk_label
"""

import os
import numpy as np
import pandas as pd

# ⚠️ ÖNEMLİ: diabet_pipeline aynı klasörde (ml) olduğu için relative import
from .diabet_pipeline import load_events, build_patient_timeseries

# Proje kök klasörü (Diyabet-AI)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Ohio event verisinin yolu (kendi dosya yolun)
CSV_EVENTS_PATH = (
    r"C:\Users\hulya\OneDrive\Masaüstü\Diyabet-AI\mühendislik projesi\data\ohio\all_patients_events.csv"
)

# Çıkacak dataset dosyası
DATA_DIR = os.path.join(ROOT_DIR, "data")
CSV_OUT_PATH = os.path.join(DATA_DIR, "risk_training_data.csv")


def build_risk_dataframe() -> pd.DataFrame:
    # Yol gerçekten var mı, önce kontrol edelim:
    if not os.path.exists(CSV_EVENTS_PATH):
        raise FileNotFoundError(
            f"Ohio events CSV bulunamadı:\n  {CSV_EVENTS_PATH}\n"
            "Yolu kontrol et, sonra scripti tekrar çalıştır."
        )

    print("👉 Event verisi yükleniyor...")
    df_all = load_events(CSV_EVENTS_PATH)

    all_rows = []

    patient_ids = df_all["patient_id"].dropna().unique()
    print(f"Bulunan hasta sayısı: {len(patient_ids)}")

    for pid in patient_ids:
        print(f"  • Hasta {pid} işleniyor...")
        g = build_patient_timeseries(
            df_all,
            patient_id=pid,
            freq="5min",
            horizon_steps=6,  # 6*5 dk = 30 dk sonrası
        )

        # Glikoz ve risk sınıfı olmayan satırları at
        g = g.dropna(subset=["glucose", "risk_class_30m"]).copy()
        if g.empty:
            continue

        # NaN'leri 0 ile doldur (feature tarafında)
        for col in [
            "carbs_1h",
            "carbs_2h",
            "carbs_4h",
            "exercise_1h",
            "exercise_intensity_mean_1h",
            "steps_30min",
            "is_sleep",
            "time_sin",
            "time_cos",
        ]:
            if col in g.columns:
                g[col] = g[col].fillna(0.0)

        for _, r in g.iterrows():
            row = {
                # ---- feature_config.XGB_FEATURE_ORDER ile aynı sıra ----
                "glucose": float(r["glucose"]),
                "carbs_win_1": float(r.get("carbs_1h", 0.0)),
                "carbs_win_2": float(r.get("carbs_2h", 0.0)),
                "carbs_win_3": float(r.get("carbs_4h", 0.0)),
                "bolus": 0.0,        # şimdilik yok
                "bolus_corr": 0.0,   # şimdilik yok
                "basal": 0.0,        # şimdilik yok
                "ex_minutes": float(r.get("exercise_1h", 0.0)),
                "ex_intensity": float(r.get("exercise_intensity_mean_1h", 0.0)),
                "steps": float(r.get("steps_30min", 0.0)),
                "is_sleep": float(r.get("is_sleep", 0.0)),
                "time_sin": float(r.get("time_sin", 0.0)),
                "time_cos": float(r.get("time_cos", 0.0)),
                # ---- hedef ----
                "risk_label": int(r["risk_class_30m"]),
            }
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("Hiç satır üretilmedi, event verisini / feature mantığını kontrol et.")

    df_out = pd.DataFrame(all_rows)
    return df_out


def main():
    # data klasörü yoksa oluştur
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    print("=== RISK DATASET EXPORT BAŞLADI ===")
    df = build_risk_dataframe()
    print("Toplam eğitim satırı:", len(df))

    cols = [
        "glucose",
        "carbs_win_1",
        "carbs_win_2",
        "carbs_win_3",
        "bolus",
        "bolus_corr",
        "basal",
        "ex_minutes",
        "ex_intensity",
        "steps",
        "is_sleep",
        "time_sin",
        "time_cos",
        "risk_label",
    ]
    df = df[cols]

    df.to_csv(CSV_OUT_PATH, index=False)
    print("✅ Dataset kaydedildi:")
    print("  ", CSV_OUT_PATH)
    print("=== EXPORT BİTTİ ===")


if __name__ == "__main__":
    main()
