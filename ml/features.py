# ml/features.py

import numpy as np
from .feature_config import XGB_FEATURE_ORDER


def build_lstm_window(
    conn,
    kullanici_id: int,
    window_size: int = 12,
    meal_window_hours: int = 3,
    exercise_window_hours: int = 6,
):
    """
    Son window_size adet ölçümden oluşan LSTM/XGB input penceresi üretir.

    Her zaman adımı (t_i) için feature'lar:
      0  - glucose            (OlcumGecmisi.Glikoz)
      1–3 - carbs_win_*       (son meal_window_hours içindeki toplam KH, 3 kere kopya)
      4  - bolus              (şimdilik 0)
      5  - bolus_corr         (şimdilik 0)
      6  - basal              (şimdilik 0)
      7  - ex_minutes         (son exercise_window_hours içindeki toplam süre)
      8  - ex_intensity       (aynı pencere içindeki MAX Seviye)
      9  - steps              (aynı pencere içindeki toplam AdimSayisi)
      10 - is_sleep           (UykuGecmisi aralığının içinde ise 1, yoksa 0)
      11 - time_sin           (günün dakikasına göre sin)
      12 - time_cos           (günün dakikasına göre cos)
    """

    cursor = conn.cursor()

    # 1) Son window_size ölçümü (en yeni en üstte)
    cursor.execute(
        """
        SELECT TOP (?) Glikoz, OlcumTarihSaat
        FROM OlcumGecmisi
        WHERE KullaniciId = ?
        ORDER BY OlcumTarihSaat DESC, Id DESC
        """,
        (window_size, kullanici_id),
    )
    rows = cursor.fetchall()

    if not rows:
        return None

    if len(rows) < 3:
        # Çok az veri, pencere üretmiyoruz
        return None

    # rows şu anda: [ (glc, ts), (glc, ts), ... ] yeni → eski
    # Kronolojik sıraya çevir (eski → yeni)
    rows = list(reversed(rows))

    # 3–11 kayıt varsa tekrar ederek doldur
    if len(rows) < window_size:
        base = rows[:]
        while len(rows) < window_size:
            rows.extend(base)
        rows = rows[-window_size:]

    # Artık tam window_size kadar ölçümümüz var
    feature_rows = []

    for glc, ts in rows:
        glc = float(glc or 0.0)

        # ---------- 2) Son meal_window_hours içindeki öğünler ----------
        cursor.execute(
            """
            SELECT ISNULL(SUM(Karbonhidrat), 0.0)
            FROM OgunGecmisi
            WHERE KullaniciId = ?
              AND OgunZamani >= DATEADD(HOUR, ?, ?)
              AND OgunZamani <= ?
            """,
            (kullanici_id, -meal_window_hours, ts, ts),
        )
        carbs_total = float(cursor.fetchone()[0] or 0.0)

        # 3 kez kopyalayalım (eski dataset'te KH_1, KH_2, KH_3 gibi düşün)
        carbs_1 = carbs_total
        carbs_2 = carbs_total
        carbs_3 = carbs_total

        # ---------- 3) Son exercise_window_hours içindeki egzersiz ----------
        cursor.execute(
            """
            SELECT
                ISNULL(SUM(SureDakika), 0.0)   AS toplam_sure,
                ISNULL(MAX(Seviye), 0.0)       AS max_seviye,
                ISNULL(SUM(AdimSayisi), 0.0)   AS toplam_adim
            FROM EgzersizGecmisi
            WHERE KullaniciId = ?
              AND EgzersizZamani >= DATEADD(HOUR, ?, ?)
              AND EgzersizZamani <= ?
            """,
            (kullanici_id, -exercise_window_hours, ts, ts),
        )
        ex_row = cursor.fetchone()
        ex_minutes = float(ex_row[0] or 0.0)
        ex_intensity = float(ex_row[1] or 0.0)
        steps = float(ex_row[2] or 0.0)

        # ---------- 4) Uyku aralığında mı? ----------
        cursor.execute(
            """
            SELECT TOP 1 1
            FROM UykuGecmisi
            WHERE KullaniciId = ?
              AND UykuBaslangic <= ?
              AND UykuBitis >= ?
            """,
            (kullanici_id, ts, ts),
        )
        is_sleep = 1.0 if cursor.fetchone() else 0.0

        # ---------- 5) Gün içi saat sin/cos ----------
        minute_of_day = ts.hour * 60 + ts.minute
        time_sin = float(
            np.sin(2 * np.pi * minute_of_day / (24 * 60))
        )
        time_cos = float(
            np.cos(2 * np.pi * minute_of_day / (24 * 60))
        )

        # Bolus / basal henüz kullanılmıyor → 0
        bolus = 0.0
        bolus_corr = 0.0
        basal = 0.0

        feat = [
            glc,            # 0: glucose
            carbs_1,        # 1
            carbs_2,        # 2
            carbs_3,        # 3
            bolus,          # 4
            bolus_corr,     # 5
            basal,          # 6
            ex_minutes,     # 7
            ex_intensity,   # 8
            steps,          # 9
            is_sleep,       # 10
            time_sin,       # 11
            time_cos,       # 12
        ]
        feature_rows.append(feat)

    window = np.array(feature_rows, dtype=float)
    return window  # shape: (window_size, len(XGB_FEATURE_ORDER))
