import numpy as np


def build_lstm_window(
    conn,
    kullanici_id: int,
    window_size: int = 12,
    meal_window_hours: int = 3,          # ana pencere (UI için)
    exercise_window_hours: int = 6,
    insulin_window_hours: int = 4,
):
    """
    Feature vektörü (13):
      0  glc
      1  carbs_1h
      2  carbs_3h
      3  carbs_6h
      4  bolus_4h
      5  reserved (0.0)
      6  basal_last
      7  ex_min_6h
      8  ex_int_max_6h
      9  steps_6h
      10 is_sleep
      11 time_sin
      12 time_cos
    """

    cursor = conn.cursor()

    # En son ölçümleri al
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

    # En az 3 ölçüm yoksa pencere kurma
    if not rows or len(rows) < 3:
        return None
    rows = list(reversed(rows))

    # window_size'tan azsa başa pad et (rows + rows yerine deterministik)
    if len(rows) < window_size:
        pad_count = window_size - len(rows)
        first = rows[0]
        rows = [first] * pad_count + rows

    # fazla geldiyse kırp
    rows = rows[-window_size:]

    feature_rows = []

    def sum_carbs(hours, ts):
        cursor.execute(
            """
            SELECT ISNULL(SUM(Karbonhidrat),0)
            FROM OgunGecmisi
            WHERE KullaniciId=?
              AND OgunZamani BETWEEN DATEADD(HOUR, ?, ?) AND ?
            """,
            (kullanici_id, -hours, ts, ts),
        )
        return float(cursor.fetchone()[0] or 0.0)

    def sum_bolus(hours, ts):
        cursor.execute(
            """
            SELECT ISNULL(SUM(Doz),0)
            FROM InsulinGecmisi
            WHERE KullaniciId=? AND Tip='bolus'
              AND UygulamaZamani BETWEEN DATEADD(HOUR, ?, ?) AND ?
            """,
            (kullanici_id, -hours, ts, ts),
        )
        return float(cursor.fetchone()[0] or 0.0)

    def last_basal(ts):
        cursor.execute(
            """
            SELECT TOP 1 Doz
            FROM InsulinGecmisi
            WHERE KullaniciId=? AND Tip='basal'
              AND UygulamaZamani <= ?
            ORDER BY UygulamaZamani DESC
            """,
            (kullanici_id, ts),
        )
        r = cursor.fetchone()
        return float(r[0]) if r else 0.0

    def ex_agg(hours, ts):
        cursor.execute(
            """
            SELECT
              ISNULL(SUM(SureDakika),0),
              ISNULL(MAX(Seviye),0),
              ISNULL(SUM(AdimSayisi),0)
            FROM EgzersizGecmisi
            WHERE KullaniciId=?
              AND EgzersizZamani BETWEEN DATEADD(HOUR, ?, ?) AND ?
            """,
            (kullanici_id, -hours, ts, ts),
        )
        ex = cursor.fetchone()
        ex_min = float(ex[0] or 0.0)
        ex_int = float(ex[1] or 0.0)
        steps = float(ex[2] or 0.0)
        return ex_min, ex_int, steps

    def is_sleeping(ts):
        cursor.execute(
            """
            SELECT 1
            FROM UykuGecmisi
            WHERE KullaniciId=?
              AND UykuBaslangic <= ?
              AND UykuBitis >= ?
            """,
            (kullanici_id, ts, ts),
        )
        return 1.0 if cursor.fetchone() else 0.0

    for glc, ts in rows:
        glc = float(glc or 0.0)

        # KH: 1h / 3h / 6h 
        carbs_1h = sum_carbs(1, ts)
        carbs_3h = sum_carbs(meal_window_hours, ts)   # default 3h
        carbs_6h = sum_carbs(6, ts)

        # bolus: son 4 saat
        bolus_4h = sum_bolus(insulin_window_hours, ts)

        # basal: son aktif basal
        basal = last_basal(ts)

        # egzersiz: son 6 saat
        ex_min, ex_int, steps = ex_agg(exercise_window_hours, ts)

        # uyku flag
        is_sleep = is_sleeping(ts)

        # zaman sin/cos
        minute = ts.hour * 60 + ts.minute
        time_sin = float(np.sin(2 * np.pi * minute / 1440.0))
        time_cos = float(np.cos(2 * np.pi * minute / 1440.0))

        feature_rows.append([
            glc,
            carbs_1h, carbs_3h, carbs_6h,
            bolus_4h,
            0.0,              
            basal,
            ex_min, ex_int, steps,
            is_sleep,
            time_sin, time_cos
        ])

    return np.array(feature_rows, dtype=float)
