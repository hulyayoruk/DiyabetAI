import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    jsonify,
    send_file,
)
import pyodbc
from datetime import timedelta, datetime
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

from ml.serve_models import predict_all, predict_xgb
from ml.features import build_lstm_window
from dotenv import load_dotenv
import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
app.permanent_session_lifetime = timedelta(minutes=30)

def get_connection():
    conn_str = os.getenv("DB_CONN_STR")
    if not conn_str:
        conn_str = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "SERVER=HLY\\MSSQLSERVER01;"
            "Database=DiyabetAI;"
            "Trusted_Connection=yes;"
        )
    return pyodbc.connect(conn_str)



def compute_glucose_trend(values, eps=3.0):
    """
    values: en güncel değer en başta olacak şekilde liste (DESC çektiğin için).
    eps: mg/dL eşiği. Çok küçük oynamaları 'stabil' say.
    """
    if not values or len(values) < 2:
        return {
            "trend": "stable",
            "trend_text": "Yeterli veri yok",
            "trend_class": "neutral",
        }

    last = float(values[0])
    prev = float(values[1])
    diff = last - prev

    if diff > eps:
        return {
            "trend": "rising",
            "trend_text": "Yükseliş eğiliminde",
            "trend_class": "up",
        }
    elif diff < -eps:
        return {
            "trend": "falling",
            "trend_text": "Düşüş eğiliminde",
            "trend_class": "down",
        }
    else:
        return {
            "trend": "stable",
            "trend_text": "Normal / stabil",
            "trend_class": "neutral",
        }


#  Dashboard (Ana Sayfa)
@app.route("/")
def index():
    if "email" not in session:
        return redirect("/giris")

    conn = get_connection()
    cursor = conn.cursor()

    # 1️⃣ Kullanıcı + Kullanıcı Bilgileri
    cursor.execute(
        """
        SELECT 
            k.AdSoyad,
            kb.Yas,
            kb.Kilo,
            kb.Boy,
            kb.TaniTarihi,
            kb.Durum,
            kb.IlacBilgileri
        FROM Kullanici k
        LEFT JOIN KullaniciBilgileri kb ON k.Id = kb.KullaniciId
        WHERE k.Email = ?
        """,
        (session["email"],),
    )
    row = cursor.fetchone()

    adsoyad = row[0] if row and row[0] else "Kullanıcı"
    yas = row[1] if row and row[1] is not None else None
    kilo = row[2] if row and row[2] is not None else None
    boy = row[3] if row and row[3] is not None else None
    tani_tarihi = row[4] if row and row[4] is not None else None
    durum = row[5] if row and row[5] is not None else None
    ilac_bilgileri = row[6] if row and row[6] is not None else None

    # 2️⃣ Kan Şekeri İstatistikleri
    cursor.execute(
        """
        SELECT TOP 30 Glikoz
        FROM OlcumGecmisi
        WHERE KullaniciId = (SELECT Id FROM Kullanici WHERE Email = ?)
        ORDER BY OlcumTarihSaat DESC, Id DESC
        """,
        (session["email"],),
    )
    rows = cursor.fetchall()

    current_glucose = None
    average_glucose = None

    if rows:
        values = [float(r[0]) for r in rows if r[0] is not None]
        if values:
            current_glucose = values[0]
            average_glucose = sum(values) / len(values)

    # ✅ 3) HbA1c tahmini (DCCT formülü)
    hba1c_est = None
    if average_glucose is not None:
        hba1c_est = (average_glucose + 46.7) / 28.7

    # 3️⃣ AI Tahmini (30 dk sonrası)
    predicted_glucose = None
    risk = "Henüz yeterli veri yok, risk tespit edilmedi."

    try:
        cursor.execute(
            """
            SELECT TOP 1 OlcumTarihSaat
            FROM OlcumGecmisi
            WHERE KullaniciId = (SELECT Id FROM Kullanici WHERE Email = ?)
            ORDER BY OlcumTarihSaat DESC, Id DESC
            """,
            (session["email"],),
        )
        last_row = cursor.fetchone()

        if last_row:
            cursor.execute(
                "SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],)
            )
            user_row = cursor.fetchone()
            if user_row:
                kullanici_id = user_row[0]
                window = build_lstm_window(conn, kullanici_id)

                if window is not None:
                    info = predict_all(window)
                    predicted_glucose = round(float(info["prediction"]), 1)

                    risk_map = {
                        0: "Hipoglisemi riski",
                        1: "Kontrol altında",
                        2: "Hiperglisemi riski",
                    }
                    rc = info.get("risk_class", 1)
                    risk = risk_map.get(rc, "Bilinmeyen durum")

    except Exception as e:
        print("❌ AI Tahmini Hatası:", e)

    conn.close()

    return render_template(
        "index.html",
        adsoyad=adsoyad,
        yas=yas,
        kilo=kilo,
        boy=boy,
        tani_tarihi=tani_tarihi,
        durum=durum,
        ilac_bilgileri=ilac_bilgileri,
        current_glucose=current_glucose,
        average_glucose=average_glucose,
        predicted_glucose=predicted_glucose,
        risk=risk,
        hba1c_est=hba1c_est,
    )


#  Yeni Ölçüm Ekleme
@app.route("/olcum_ekle", methods=["POST"])
def olcum_ekle():
    if "email" not in session:
        return redirect("/giris")

    glikoz = request.form.get("glikoz")
    ilac = request.form.get("ilac")
    notlar = request.form.get("notlar")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = user[0]
    now = datetime.now()

    cursor.execute(
        """
        INSERT INTO OlcumGecmisi
            (KullaniciId, OlcumTarihSaat, Glikoz, Ilac, Notlar)
        VALUES (?, ?, ?, ?, ?)
        """,
        (kullanici_id, now, glikoz, ilac, notlar),
    )

    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/ogun_ekle", methods=["POST"])
def ogun_ekle():
    if "email" not in session:
        return redirect("/giris")

    ogun_zaman_str = request.form.get("ogun_zaman")
    ogun_turu = request.form.get("ogun_turu") or None
    karbonhidrat = request.form.get("karbonhidrat") or None
    notlar = request.form.get("ogun_not") or None

    try:
        ogun_zamani = datetime.fromisoformat(ogun_zaman_str) if ogun_zaman_str else datetime.now()
    except Exception:
        ogun_zamani = datetime.now()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = user[0]

    cursor.execute(
        """
        INSERT INTO OgunGecmisi (KullaniciId, OgunZamani, OgunTuru, Karbonhidrat, Notlar)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            kullanici_id,
            ogun_zamani,
            ogun_turu,
            float(karbonhidrat) if karbonhidrat else None,
            notlar,
        ),
    )

    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/egzersiz_ekle", methods=["POST"])
def egzersiz_ekle():
    if "email" not in session:
        return redirect("/giris")

    eg_zaman_str = request.form.get("egzersiz_zaman")
    sure = request.form.get("egzersiz_suresi") or None
    seviye = request.form.get("egzersiz_seviyesi") or None
    adim = request.form.get("adim_sayisi") or None
    notlar = request.form.get("egzersiz_not") or None

    try:
        eg_zamani = datetime.fromisoformat(eg_zaman_str) if eg_zaman_str else datetime.now()
    except Exception:
        eg_zamani = datetime.now()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = user[0]

    cursor.execute(
        """
        INSERT INTO EgzersizGecmisi (KullaniciId, EgzersizZamani, SureDakika, Seviye, AdimSayisi, Notlar)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            kullanici_id,
            eg_zamani,
            int(sure) if sure else None,
            int(seviye) if seviye else None,
            int(adim) if adim else None,
            notlar,
        ),
    )

    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/uyku_ekle", methods=["POST"])
def uyku_ekle():
    if "email" not in session:
        return redirect("/giris")

    uyku_baslangic_str = request.form.get("uyku_baslangic")
    uyku_bitis_str = request.form.get("uyku_bitis")
    notlar = request.form.get("uyku_not") or None

    try:
        uyku_baslangic = datetime.fromisoformat(uyku_baslangic_str) if uyku_baslangic_str else None
    except Exception:
        uyku_baslangic = None

    try:
        uyku_bitis = datetime.fromisoformat(uyku_bitis_str) if uyku_bitis_str else None
    except Exception:
        uyku_bitis = None

    toplam_dakika = None
    if uyku_baslangic and uyku_bitis:
        diff = uyku_bitis - uyku_baslangic
        toplam_dakika = int(diff.total_seconds() // 60)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = user[0]

    cursor.execute(
        """
        INSERT INTO UykuGecmisi (KullaniciId, UykuBaslangic, UykuBitis, ToplamDakika, Notlar)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            kullanici_id,
            uyku_baslangic,
            uyku_bitis,
            toplam_dakika,
            notlar,
        ),
    )

    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/api/activity_summary")
def activity_summary():
    if "email" not in session:
        return jsonify({"error": "Yetkisiz erişim"}), 401

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "Kullanıcı bulunamadı"}), 400

    kullanici_id = user[0]

    cursor.execute(
        """
        SELECT ISNULL(SUM(Karbonhidrat), 0)
        FROM OgunGecmisi
        WHERE KullaniciId = ?
          AND OgunZamani >= DATEADD(day, -1, GETDATE())
        """,
        (kullanici_id,),
    )
    row_ogun = cursor.fetchone()
    total_carb = float(row_ogun[0] or 0)

    cursor.execute(
        """
        SELECT
            ISNULL(SUM(SureDakika), 0)    AS total_ex_minutes,
            ISNULL(AVG(Seviye), 0)        AS avg_intensity,
            ISNULL(SUM(AdimSayisi), 0)    AS total_steps
        FROM EgzersizGecmisi
        WHERE KullaniciId = ?
          AND EgzersizZamani >= DATEADD(day, -1, GETDATE())
        """,
        (kullanici_id,),
    )
    row_ex = cursor.fetchone()
    total_ex_minutes = float(row_ex[0] or 0)
    avg_intensity = float(row_ex[1] or 0)
    total_steps = int(row_ex[2] or 0)

    cursor.execute(
        """
        SELECT ISNULL(SUM(ToplamDakika), 0)
        FROM UykuGecmisi
        WHERE KullaniciId = ?
          AND UykuBaslangic >= DATEADD(day, -1, GETDATE())
        """,
        (kullanici_id,),
    )
    row_sleep = cursor.fetchone()
    total_sleep = float(row_sleep[0] or 0)

    conn.close()

    return jsonify(
        {
            "total_carb": total_carb,
            "total_ex_minutes": total_ex_minutes,
            "avg_intensity": avg_intensity,
            "total_steps": total_steps,
            "total_sleep": total_sleep,
        }
    )


#  Kayıt Ol
@app.route("/kayit", methods=["GET", "POST"])
def kayit():
    if request.method == "GET":
        return render_template("kayit.html", hata=None)

    adsoyad = (request.form.get("adsoyad") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    sifre = request.form.get("sifre") or ""
    sifre_tekrar = request.form.get("sifre_tekrar") or ""
    yas = request.form.get("yas") or None
    kilo = request.form.get("kilo") or None
    boy = request.form.get("boy") or None
    tanitarihi = request.form.get("tanitarihi") or None
    durum = request.form.get("durum") or None

    if sifre != sifre_tekrar:
        return render_template("kayit.html", hata="Şifreler uyuşmuyor!")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Id FROM Kullanici WHERE LOWER(Email) = ?", (email,))
    exists = cursor.fetchone()
    if exists:
        conn.close()
        return render_template("kayit.html", hata="Bu e-posta ile zaten bir hesap var.")

    sifre_hashed = generate_password_hash(sifre)

    cursor.execute(
        """
        INSERT INTO Kullanici (AdSoyad, Email, Sifre)
        VALUES (?, ?, ?)
        """,
        (adsoyad, email, sifre_hashed),
    )

    cursor.execute("SELECT Id FROM Kullanici WHERE LOWER(Email) = ?", (email,))
    row = cursor.fetchone()
    if not row:
        conn.rollback()
        conn.close()
        return render_template("kayit.html", hata="Kayıt sırasında bir hata oluştu.")

    kullanici_id = row[0]

    yas_val = int(yas) if yas else None
    kilo_val = float(kilo) if kilo else None
    boy_val = int(boy) if boy else None
    tani_val = tanitarihi if tanitarihi else None

    cursor.execute(
        """
        INSERT INTO KullaniciBilgileri
            (KullaniciId, Boy, TaniTarihi, Yas, Kilo, Durum, IlacBilgileri)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (kullanici_id, boy_val, tani_val, yas_val, kilo_val, durum, None),
    )

    conn.commit()
    conn.close()

    session.permanent = True
    session["email"] = email
    return redirect("/")


#  Giriş Yap
@app.route("/giris", methods=["GET", "POST"])
def giris():
    if request.method == "GET":
        return render_template("giris.html", hata=None)

    email = (request.form.get("email") or "").strip().lower()
    sifre = request.form.get("sifre") or ""

    if not email or not sifre:
        return render_template("giris.html", hata="Lütfen e-posta ve şifre girin.")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Email, Sifre FROM Kullanici WHERE LOWER(Email) = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return render_template("giris.html", hata="E-posta veya şifre hatalı!")

    stored_email = user[0]
    stored_pwd = user[1] or ""

    ok = False
    try:
        ok = check_password_hash(stored_pwd, sifre)
    except Exception:
        ok = False

    if not ok and not stored_pwd.startswith("pbkdf2:"):
        ok = (stored_pwd == sifre)

    if not ok:
        return render_template("giris.html", hata="E-posta veya şifre hatalı!")

    session.permanent = True
    session["email"] = stored_email.lower()
    return redirect("/")


@app.route("/profil", methods=["GET", "POST"])
def profil_panel():
    if "email" not in session:
        return redirect("/giris")

    conn = get_connection()
    cursor = conn.cursor()
    mesaj = None

    if request.method == "POST":
        adsoyad = request.form.get("adsoyad")
        yas = request.form.get("yas")
        kilo = request.form.get("kilo")
        boy = request.form.get("boy")
        durum = request.form.get("durum")
        tani_tarihi = request.form.get("tani_tarihi")
        ilac_bilgileri = request.form.get("ilac_bilgileri")

        boy_val = int(boy) if boy else None
        yas_val = int(yas) if yas else None
        kilo_val = float(kilo) if kilo else None
        tani_val = tani_tarihi if tani_tarihi else None

        cursor.execute(
            """
            UPDATE Kullanici
            SET AdSoyad = ?
            WHERE Email = ?
            """,
            (adsoyad, session["email"]),
        )

        cursor.execute(
            """
            UPDATE KullaniciBilgileri
            SET Boy = ?, 
                TaniTarihi = ?,
                Yas = ?,
                Kilo = ?,
                Durum = ?,
                IlacBilgileri = ?
            WHERE KullaniciId = (SELECT Id FROM Kullanici WHERE Email = ?)
            """,
            (boy_val, tani_val, yas_val, kilo_val, durum, ilac_bilgileri, session["email"]),
        )

        conn.commit()
        mesaj = "Profil bilgileri başarıyla güncellendi ✅"

    cursor.execute(
        """
        SELECT 
            k.AdSoyad,
            kb.Boy,
            kb.TaniTarihi,
            kb.Yas,
            kb.Kilo,
            kb.Durum,
            kb.IlacBilgileri
        FROM Kullanici k
        LEFT JOIN KullaniciBilgileri kb ON k.Id = kb.KullaniciId
        WHERE k.Email = ?
        """,
        (session["email"],),
    )
    row = cursor.fetchone()
    conn.close()

    return render_template(
        "profil.html",
        mesaj=mesaj,
        adsoyad=row[0] if row else "",
        boy=row[1] if row else None,
        tanitarihi=row[2] if row else None,
        yas=row[3] if row else None,
        kilo=row[4] if row else None,
        durum=row[5] if row else None,
        ilac_bilgileri=row[6] if row else "",
    )


@app.route("/sifremi_unuttum", methods=["GET", "POST"])
def sifremi_unuttum():
    if request.method == "POST":
        email = request.form["email"].strip().lower()

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Id FROM Kullanici WHERE LOWER(Email)=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            import secrets
            token = secrets.token_urlsafe(16)
            session["reset_email"] = email
            session["reset_token"] = token
            return redirect("/sifremi_unuttum_ok")

        return render_template("sifremi_unuttum.html", hata="Bu e-posta sistemde kayıtlı değil!")

    return render_template("sifremi_unuttum.html")


@app.route("/sifremi_unuttum_ok")
def sifremi_unuttum_ok():
    return render_template("sifremi_unuttum_ok.html")


@app.route("/sifre_sifirla/<token>", methods=["GET", "POST"])
def sifre_sifirla(token):
    if "reset_token" not in session or session["reset_token"] != token:
        return redirect("/giris")

    if request.method == "POST":
        yeni_sifre = request.form["sifre"]
        yeni_hash = generate_password_hash(yeni_sifre)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE Kullanici SET Sifre=? WHERE Email=?", (yeni_hash, session["reset_email"]))
        conn.commit()
        conn.close()

        session.pop("reset_token", None)
        session.pop("reset_email", None)
        return redirect("/giris")

    return render_template("sifre_sifirla.html", token=token)


@app.route("/cikis")
def cikis():
    session.clear()
    return redirect("/giris")


# =========================================
# ✅ API - Ölçüm Geçmişi (grafik + trend için)
# =========================================
@app.route("/api/data")
def get_data():
    if "email" not in session:
        return jsonify({"error": "Yetkisiz erişim"}), 401

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT TOP 10 OlcumTarihSaat, Glikoz
        FROM OlcumGecmisi
        WHERE KullaniciId = (SELECT Id FROM Kullanici WHERE Email = ?)
        ORDER BY OlcumTarihSaat DESC, Id DESC
        """,
        (session["email"],),
    )
    rows = cursor.fetchall()
    conn.close()

    # values DESC (en güncel en başta)
    values_desc = [float(r[1]) for r in rows if r[1] is not None]

    def compute_glucose_trend(values, eps=3.0):
        if not values or len(values) < 2:
            return {"trend": "stable", "trend_text": "Yeterli veri yok", "trend_class": "neutral"}

        last = float(values[0])
        prev = float(values[1])
        diff = last - prev

        if diff > eps:
            return {"trend": "rising", "trend_text": "Değerler yükseliş eğiliminde", "trend_class": "up"}
        elif diff < -eps:
            return {"trend": "falling", "trend_text": "Değerler düşüş eğiliminde", "trend_class": "down"}
        else:
            return {"trend": "stable", "trend_text": "Normal / stabil", "trend_class": "neutral"}

    trend_info = compute_glucose_trend(values_desc, eps=3.0)

    # Chart için ters çevir (eski -> yeni)
    labels = [r[0].strftime("%d.%m %H:%M") for r in rows][::-1]
    values = [r[1] for r in rows][::-1]

    return jsonify(
        {
            "labels": labels,
            "values": values,
            "trend": trend_info["trend"],
            "trend_text": trend_info["trend_text"],
            "trend_class": trend_info["trend_class"],
        }
    )

@app.route("/api/ai_suggestions")
def ai_suggestions():
    if "email" not in session:
        return jsonify({"error": "login required"}), 401

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "user not found"}), 400

        kullanici_id = row[0]

        window = build_lstm_window(conn, kullanici_id)
        if window is None:
            return jsonify({"error": "not_enough_data"}), 400

        info = predict_all(window)
        base_feat = window[-1].copy()

        # --- base prediction & probs ---
        prediction_val = float(info["prediction"])

        base_probs = predict_xgb(base_feat.reshape(1, -1))[0]
        base_p_hypo = float(base_probs[0])
        base_p_normal = float(base_probs[1])
        base_p_hyper = float(base_probs[2])

        def probs_to_dict(probs):
            return {
                "p_hypo": float(probs[0]) * 100.0,
                "p_normal": float(probs[1]) * 100.0,
                "p_hyper": float(probs[2]) * 100.0,
            }

        base_prob_dict = probs_to_dict(base_probs)

        # --- night hypo risk ---
        is_sleep_now = bool(base_feat[10] >= 0.5)
        now_hour = datetime.now().hour
        is_night = (now_hour >= 23 or now_hour < 7) or is_sleep_now
        night_hypo_risk_pct = (base_p_hypo * 100.0) if is_night else 0.0

        # --- insulin context ---
        last_basal_dose = 0.0
        bolus_6h = 0.0
        current_dose = 0.0
        current_dose_type = None

        try:
            cursor.execute(
                """
                SELECT TOP 1 Tip, Doz
                FROM InsulinGecmisi
                WHERE KullaniciId=?
                ORDER BY UygulamaZamani DESC
                """,
                (kullanici_id,),
            )
            last_any = cursor.fetchone()
            if last_any:
                current_dose_type = last_any[0]
                current_dose = float(last_any[1] or 0.0)
        except Exception:
            current_dose_type = None
            current_dose = 0.0

        try:
            cursor.execute(
                """
                SELECT TOP 1 Doz, UygulamaZamani
                FROM InsulinGecmisi
                WHERE KullaniciId=? AND Tip='basal'
                ORDER BY UygulamaZamani DESC
                """,
                (kullanici_id,),
            )
            last_basal = cursor.fetchone()
            last_basal_dose = float(last_basal[0]) if last_basal else 0.0

            cursor.execute(
                """
                SELECT ISNULL(SUM(Doz),0)
                FROM InsulinGecmisi
                WHERE KullaniciId=? AND Tip='bolus'
                  AND UygulamaZamani >= DATEADD(HOUR, -6, GETDATE())
                """,
                (kullanici_id,),
            )
            bolus_6h = float(cursor.fetchone()[0] or 0.0)
        except Exception:
            last_basal_dose = 0.0
            bolus_6h = 0.0

        TH_HYPO_WARN = 0.20
        TH_HYPO_HIGH = 0.35

        hypo_warning = None
        if is_night:
            if base_p_hypo >= TH_HYPO_HIGH:
                hypo_warning = {
                    "level": "high",
                    "title": "Gece hipoglisemi riski yüksek",
                    "text": (
                        f"Model bu koşullarda hipoglisemi olasılığını %{night_hypo_risk_pct:.1f} görüyor. "
                        f"(Son basal: {last_basal_dose:.1f}U, son 6s bolus toplamı: {bolus_6h:.1f}U)"
                    ),
                }
            elif base_p_hypo >= TH_HYPO_WARN:
                hypo_warning = {
                    "level": "warn",
                    "title": "Gece hipoglisemi riski artmış olabilir",
                    "text": (
                        f"Model hipoglisemi olasılığını %{night_hypo_risk_pct:.1f} görüyor. "
                        f"(Son basal: {last_basal_dose:.1f}U, son 6s bolus toplamı: {bolus_6h:.1f}U)"
                    ),
                }

        # --- UI context from DB ---
        cursor.execute(
            """
            SELECT TOP 1 Karbonhidrat
            FROM OgunGecmisi
            WHERE KullaniciId = ? AND Karbonhidrat > 0
            ORDER BY OgunZamani DESC
            """,
            (kullanici_id,),
        )
        last_carb_row = cursor.fetchone()
        current_carbs = float(last_carb_row[0]) if last_carb_row else 0.0

        current_ex_min = float(base_feat[7])

        # --- helper: changed indices ---
        def diff_indices(a, b, eps=1e-9, limit=50):
            out = []
            for i in range(len(a)):
                if abs(float(a[i]) - float(b[i])) > eps:
                    out.append(i)
                    if len(out) >= limit:
                        break
            return out

        # -------------------------
        # SIM CARDS (aynı kalsın)
        # -------------------------
        simulations = []

        if current_carbs > 0:
            feat_lowcarb = base_feat.copy()
            feat_lowcarb[1:4] = feat_lowcarb[1:4] * 0.8
            probs_lowcarb = predict_xgb(feat_lowcarb.reshape(1, -1))[0]
            simulations.append(
                {
                    "title": "-%20 karbonhidrat",
                    "subtitle": f"Mevcut KH ~ {current_carbs:.1f} g",
                    "before": base_prob_dict,
                    "after": probs_to_dict(probs_lowcarb),
                }
            )

        if current_ex_min >= 0:
            feat_moreex = base_feat.copy()
            feat_moreex[7] = feat_moreex[7] * 2.0
            feat_moreex[9] = feat_moreex[9] + 4000
            probs_moreex = predict_xgb(feat_moreex.reshape(1, -1))[0]
            simulations.append(
                {
                    "title": "+%50 ek egzersiz süresi",
                    "subtitle": f"Mevcut süre ~ {current_ex_min:.1f} dk",
                    "before": base_prob_dict,
                    "after": probs_to_dict(probs_moreex),
                }
            )

        # -------------------------
        # SWEEP TEST (neden değişmiyor?)
        # - KH ve egzersizi daha geniş aralıkta oynat
        # -------------------------
        sweep = {"carb_multipliers": [], "ex_multipliers": []}

        # KH sweep (1:4 birlikte)
        for m in [0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0]:
            f = base_feat.copy()
            f[1:4] = f[1:4] * m
            p = predict_xgb(f.reshape(1, -1))[0]
            sweep["carb_multipliers"].append(
                {
                    "multiplier": m,
                    "changed_indices": diff_indices(base_feat, f),
                    "after": probs_to_dict(p),
                }
            )

        # Egzersiz sweep (index 7)
        for m in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            f = base_feat.copy()
            f[7] = f[7] * m
            p = predict_xgb(f.reshape(1, -1))[0]
            sweep["ex_multipliers"].append(
                {
                    "multiplier": m,
                    "changed_indices": diff_indices(base_feat, f),
                    "after": probs_to_dict(p),
                }
            )

        # -------------------------
        # Risk label/desc
        # -------------------------
        risk_label = (
            "Hipoglisemi riski"
            if info["risk_class"] == 0
            else "Kontrol altında"
            if info["risk_class"] == 1
            else "Hiperglisemi riski"
        )

        risk_desc = (
            "Kan şekerin düşebilir, dikkatli ol."
            if info["risk_class"] == 0
            else "30 dk içinde değerler normal aralıkta görünüyor."
            if info["risk_class"] == 1
            else "Kan şekerin yükselebilir, yakın takipte kal."
        )

        trend = info.get("trend")
        trend_desc = (
            "Değerler artış eğiliminde 📈"
            if trend == "rising"
            else "Değerler düşüş eğiliminde 📉"
            if trend == "falling"
            else "Şu an sabit bir seyir var ⚖️"
        )

        # -------------------------
        # DEBUG
        # -------------------------
        debug = {
            "base_feat_head": [float(x) for x in base_feat[:12]],
            "note": (
                "Eğer sweep'te de olasılıklar hiç değişmiyorsa, XGB bu feature'ları kullanmıyor "
                "veya aynı leaf'te kalıyorsunuz. Eğer sadece bazı multiplier'larda değişiyorsa, "
                "threshold etkisi var."
            ),
        }

        return jsonify(
            {
                "prediction": prediction_val,
                "trend": trend,
                "trend_desc": trend_desc,
                "risk_label": risk_label,
                "risk_desc": risk_desc,

                "simulations": simulations,
                "sweep": sweep,       # ✅ yeni
                "debug": debug,       # ✅ sade debug

                "p_hypo": base_p_hypo * 100,
                "p_normal": base_p_normal * 100,
                "p_hyper": base_p_hyper * 100,

                "night_hypo_risk_pct": night_hypo_risk_pct,
                "is_night": is_night,
                "hypo_warning": hypo_warning,
                "insulin_context": {
                    "last_basal_dose": last_basal_dose,
                    "bolus_last_6h": bolus_6h,
                    "current_insulin_dose": current_dose,
                    "current_insulin_type": current_dose_type,
                },
            }
        )

    except Exception as e:
        print("[AI Error]", e)
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

@app.route("/insulin_ekle", methods=["POST"])
def insulin_ekle():
    if "email" not in session:
        return redirect("/giris")

    tip = request.form["tip"]
    doz = float(request.form["doz"])
    zaman = request.form.get("zaman")
    zaman = datetime.fromisoformat(zaman) if zaman else datetime.now()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email=?", (session["email"],))
    uid = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO InsulinGecmisi (KullaniciId, Tip, Doz, UygulamaZamani)
        VALUES (?, ?, ?, ?)
        """,
        (uid, tip, doz, zaman),
    )

    conn.commit()
    conn.close()
    return redirect("/")


@app.route("/api/period_report")
def period_report():
    if "email" not in session:
        return jsonify({"error": "Yetkisiz erişim"}), 401

    days_str = request.args.get("days", "7")
    try:
        days = int(days_str)
    except ValueError:
        days = 7

    if days not in (7, 15, 30, 90):
        days = 7

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Kullanıcı bulunamadı"}), 400

    kullanici_id = row[0]

    now = datetime.now()
    start_cur = now - timedelta(days=days)
    end_cur = now

    start_prev = now - timedelta(days=2 * days)
    end_prev = start_cur

    def get_stats(start, end):
        cursor.execute(
            """
            SELECT Glikoz
            FROM OlcumGecmisi
            WHERE KullaniciId = ?
              AND OlcumTarihSaat >= ?
              AND OlcumTarihSaat < ?
            """,
            (kullanici_id, start, end),
        )
        rows = cursor.fetchall()
        values = [float(r[0]) for r in rows if r[0] is not None]

        if not values:
            return None

        n = len(values)
        avg_val = sum(values) / n

        hypo = len([v for v in values if v < 70]) / n * 100
        inrange = len([v for v in values if 70 <= v <= 180]) / n * 100
        hyper = len([v for v in values if v > 180]) / n * 100

        return {"count": n, "avg": avg_val, "hypo": hypo, "inrange": inrange, "hyper": hyper}

    cur_stats = get_stats(start_cur, end_cur)
    prev_stats = get_stats(start_prev, end_prev)

    conn.close()

    if cur_stats is None:
        return jsonify({"error": "not_enough_data"}), 400

    prev_avg = prev_stats["avg"] if prev_stats is not None else None

    delta = None
    direction = "none"
    if prev_avg is not None:
        delta = cur_stats["avg"] - prev_avg
        if delta > 5:
            direction = "up"
        elif delta < -5:
            direction = "down"
        else:
            direction = "same"

    return jsonify(
        {
            "period_days": days,
            "count": cur_stats["count"],
            "avg_glucose": cur_stats["avg"],
            "hypo_pct": cur_stats["hypo"],
            "inrange_pct": cur_stats["inrange"],
            "hyper_pct": cur_stats["hyper"],
            "prev_avg_glucose": prev_avg,
            "delta_avg": delta,
            "delta_direction": direction,
        }
    )


@app.route("/api/measurement_list")
def measurement_list():
    if "email" not in session:
        return jsonify({"error": "Yetkisiz erişim"}), 401

    days_str = request.args.get("days", "30")
    try:
        days = int(days_str)
    except ValueError:
        days = 30

    if days not in (15, 30, 90):
        days = 30

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Kullanıcı bulunamadı"}), 400

    kullanici_id = row[0]

    cursor.execute(
        """
        SELECT OlcumTarihSaat, Glikoz, Ilac, Notlar
        FROM OlcumGecmisi
        WHERE KullaniciId = ?
          AND OlcumTarihSaat >= DATEADD(day, -?, GETDATE())
        ORDER BY OlcumTarihSaat DESC
        """,
        (kullanici_id, days),
    )
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        dt = r[0]
        gl = r[1]
        ilac = r[2]
        note = r[3]
        data.append(
            {
                "datetime": dt.strftime("%d.%m.%Y %H:%M") if dt else "",
                "glucose": float(gl) if gl is not None else None,
                "drug": ilac or "",
                "note": note or "",
            }
        )

    return jsonify({"rows": data})


@app.route("/rapor/olcumler/excel")
def export_measurements_excel():
    if "email" not in session:
        return "Yetkisiz erişim", 401

    days_str = request.args.get("days", "30")
    try:
        days = int(days_str)
    except ValueError:
        days = 30
    if days not in (15, 30, 90):
        days = 30

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = row[0]

    cursor.execute(
        """
        SELECT OlcumTarihSaat, Glikoz, Ilac, Notlar
        FROM OlcumGecmisi
        WHERE KullaniciId = ?
          AND OlcumTarihSaat >= DATEADD(day, -?, GETDATE())
        ORDER BY OlcumTarihSaat DESC
        """,
        (kullanici_id, days),
    )
    rows = cursor.fetchall()
    conn.close()

    data = [
        {
            "Tarih / Saat": r[0].strftime("%d.%m.%Y %H:%M") if r[0] else "",
            "Glikoz (mg/dL)": float(r[1]) if r[1] is not None else None,
            "İlaç": r[2] or "",
            "Not": r[3] or "",
        }
        for r in rows
    ]

    df = pd.DataFrame(data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ölçümler")
    output.seek(0)

    filename = f"olcum_raporu_{days}gun.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/rapor/olcumler/pdf")
def export_measurements_pdf():
    if "email" not in session:
        return "Yetkisiz erişim", 401

    days_str = request.args.get("days", "30")
    try:
        days = int(days_str)
    except ValueError:
        days = 30
    if days not in (15, 30, 90):
        days = 30

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT Id FROM Kullanici WHERE Email = ?", (session["email"],))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return "Kullanıcı bulunamadı", 400

    kullanici_id = row[0]

    cursor.execute(
        """
        SELECT OlcumTarihSaat, Glikoz, Ilac, Notlar
        FROM OlcumGecmisi
        WHERE KullaniciId = ?
          AND OlcumTarihSaat >= DATEADD(day, -?, GETDATE())
        ORDER BY OlcumTarihSaat DESC
        """,
        (kullanici_id, days),
    )
    rows = cursor.fetchall()
    conn.close()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    title = f"Son {days} güne ait ölçüm raporu"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, title)

    c.setFont("Helvetica", 9)
    y = height - 70

    c.drawString(40, y, "Tarih / Saat")
    c.drawString(150, y, "Glikoz (mg/dL)")
    c.drawString(240, y, "İlaç")
    c.drawString(360, y, "Not")
    y -= 15

    for r in rows:
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - 40

        dt_str = r[0].strftime("%d.%m.%Y %H:%M") if r[0] else ""
        gl = f"{float(r[1]):.1f}" if r[1] is not None else "-"
        ilac = r[2] or "-"
        note = r[3] or "-"

        c.drawString(40, y, dt_str)
        c.drawString(150, y, gl)
        c.drawString(240, y, (ilac[:30] + "...") if len(ilac) > 33 else ilac)
        c.drawString(360, y, (note[:40] + "...") if len(note) > 43 else note)
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)

    filename = f"olcum_raporu_{days}gun.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
