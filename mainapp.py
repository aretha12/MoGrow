import streamlit as st
import numpy as np
import joblib

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Sistem Prediksi Stunting & Kesehatan Ibu",
    layout="centered"
)

st.title("🩺 Sistem Prediksi Stunting Anak & Risiko Kesehatan Ibu")

st.markdown(
    "**Aplikasi ini menyediakan dua analisis utama:**  \n"
    "1️⃣ Prediksi Stunting Pada Anak (Hybrid ML + Rule-based)  \n"
    "2️⃣ Prediksi Risiko Kesehatan Ibu (Machine Learning)"
)

# ======================================================
# LOAD MODEL
# ======================================================
anak_model = joblib.load("bayi_random_forest.pkl")
anak_scaler = joblib.load("bayi_scaler.pkl")
AKURASI_ANAK = 0.865

ibu_model = joblib.load("ibu_random_forest.pkl")
ibu_scaler = joblib.load("ibu_scaler.pkl")
AKURASI_IBU = 0.8473

# ======================================================
# SIDEBAR MENU
# ======================================================
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["👶🏻 Prediksi Stunting Anak", "🤰🏻 Prediksi Risiko Kesehatan Ibu"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed with 💖 using Streamlit")

# ======================================================
# 👶🏻 MENU 1 – PREDIKSI STUNTING ANAK
# ======================================================
if menu == "👶🏻 Prediksi Stunting Anak":

    st.header("👶🏻 Prediksi Stunting Anak")

    st.markdown("""
    Prediksi dilakukan menggunakan **machine learning (Random Forest)**  
    dan **aturan pertumbuhan (rule-based override)** untuk mencegah
    kesalahan prediksi pada anak dengan pertumbuhan normal.
    """)

    # -----------------------------
    # INPUT DATA ANAK
    # -----------------------------
    gender_map = {"Laki-laki": 0, "Perempuan": 1}
    gender = st.selectbox("Jenis Kelamin", list(gender_map.keys()))

    age = st.number_input("Usia Anak (bulan)", 0, 60, 12)
    birth_weight = st.number_input("Berat Lahir (kg)", 0.5, 5.0, 3.0)
    birth_length = st.number_input("Panjang Lahir (cm)", 30, 60, 49)
    body_weight = st.number_input("Berat Badan Saat Ini (kg)", 1.0, 25.0, 10.0)
    body_length = st.number_input("Tinggi Badan Saat Ini (cm)", 40, 130, 70)
    breastfeeding = st.selectbox("ASI Eksklusif?", ["Ya", "Tidak"])

    g = gender_map[gender]
    bf = 1 if breastfeeding == "Ya" else 0

    # -----------------------------
    # PREDIKSI ANAK
    # -----------------------------
    if st.button("🔍 Prediksi Stunting Anak"):

        data = np.array([[g, age, birth_weight, birth_length, body_weight, body_length, bf]])
        data_scaled = anak_scaler.transform(data)
        pred_model = anak_model.predict(data_scaled)[0]

        # RULE-BASED OVERRIDE (TIDAK DIPAKSAKAN)
        is_override_normal = (
            age >= 12 and
            body_length >= 75 and
            body_weight >= 9
        )

        if is_override_normal:
            final_pred = 0
            decision_source = "Rule-based override (pertumbuhan sesuai usia)"
        else:
            final_pred = pred_model
            decision_source = "Prediksi model machine learning"

        # -----------------------------
        # OUTPUT ANAK
        # -----------------------------
        st.metric("Akurasi Model (Validasi)", f"{AKURASI_ANAK*100:.2f}%")

        if final_pred == 1:
            st.error("⚠️ Anak **terindikasi stunting**.")
        else:
            st.success("✅ Anak **tidak stunting**.")

        st.caption(f"Sumber keputusan: **{decision_source}**")

        # EDUKASI TAMBAHAN (TIDAK MEMAKSA)
        st.info(
            "Catatan: Stunting ditentukan terutama oleh **tinggi badan terhadap usia**, "
            "bukan oleh berat badan saja."
        )

        # -----------------------------
        # REKOMENDASI ANAK
        # -----------------------------
        st.subheader("🩺 Rekomendasi Kesehatan Anak")

        if final_pred == 1:
            st.markdown(f"""
            **Analisis:**  
            Anak usia **{age} bulan** dengan tinggi **{body_length} cm**
            menunjukkan risiko stunting.

            **Kemungkinan penyebab:**
            - Asupan gizi tidak seimbang  
            - Berat lahir rendah  
            - Tidak mendapat ASI eksklusif  
            - Sanitasi lingkungan kurang baik  

            **Rekomendasi:**
            1. Tingkatkan asupan protein (telur, ikan, ayam, tempe, tahu)  
            2. Perbanyak buah dan sayur  
            3. Pantau pertumbuhan rutin di posyandu  
            4. Jaga kebersihan dan sanitasi rumah  
            5. Konsultasi tenaga medis  

            💡 *Pemantauan rutin sangat penting pada 1000 HPK.*
            """)
        else:
            st.markdown(f"""
            **Analisis:**  
            Pertumbuhan anak **normal** untuk usia **{age} bulan**.

            **Saran menjaga pertumbuhan optimal:**
            - Konsumsi makanan bergizi seimbang  
            - Batasi makanan manis & instan  
            - Stimulasi perkembangan (bermain, membaca, berbicara)  
            - Tidur cukup (10–12 jam per hari)  
            - Pemeriksaan rutin di posyandu  

            💡 *Pertumbuhan optimal dipengaruhi gizi, stimulasi, dan pola asuh.*
            """)

        # -----------------------------
        # DETAIL INPUT ANAK
        # -----------------------------
        st.subheader("📊 Detail Input Anak")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usia", f"{age} bulan")
        col2.metric("Berat Badan", f"{body_weight} kg")
        col3.metric("Tinggi Badan", f"{body_length} cm")

    st.markdown("---")
    st.caption(
        "⚠️ Aplikasi ini hanya alat bantu skrining awal dan "
        "**tidak menggantikan diagnosis dokter**."
    )

# ======================================================
# 🤰🏻 MENU 2 – PREDIKSI RISIKO KESEHATAN IBU
# ======================================================
elif menu == "🤰🏻 Prediksi Risiko Kesehatan Ibu":

    st.header("🤰🏻 Prediksi Risiko Kesehatan Ibu")

    st.markdown("""
    Prediksi risiko kesehatan ibu dilakukan berdasarkan **parameter klinis**
    seperti tekanan darah, kadar gula darah, suhu tubuh, dan denyut jantung.

    Hasil prediksi digunakan sebagai **skrining awal**, bukan diagnosis medis.
    """)

    # -----------------------------
    # INPUT DATA IBU (CELSIUS)
    # -----------------------------
    age = st.number_input("Usia Ibu (tahun)", 15, 50, 28)
    sys = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
    dia = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 80)
    bs = st.number_input("Blood Sugar", 1.0, 30.0, 7.0)

    temp_c = st.number_input("Temperatur Tubuh (°C)", 30.0, 45.0, 36.5)
    temp_f = (temp_c * 9 / 5) + 32  # KONVERSI KE FAHRENHEIT

    heart = st.number_input("Heart Rate (bpm)", 50, 200, 100)

    # -----------------------------
    # PREDIKSI IBU
    # -----------------------------
    if st.button("🔍 Prediksi Risiko Ibu"):

        data = np.array([[age, sys, dia, bs, temp_f, heart]])
        data_scaled = ibu_scaler.transform(data)
        pred = ibu_model.predict(data_scaled)[0]

        st.metric("Akurasi Model (Validasi)", f"{AKURASI_IBU*100:.2f}%")

        if pred == 0:
            st.success("🟢 Risiko Rendah")
            st.caption("Kondisi vital ibu berada dalam batas aman.")
        elif pred == 1:
            st.warning("🟡 Risiko Sedang")
            st.caption("Terdapat beberapa indikator yang perlu dipantau.")
        else:
            st.error("🔴 Risiko Tinggi")
            st.caption("Parameter vital menunjukkan potensi risiko serius.")

        # -----------------------------
        # REKOMENDASI IBU
        # -----------------------------
        st.subheader("🩺 Saran Kesehatan Ibu")

        if pred == 0:
            st.markdown("""
            - Konsumsi makanan seimbang  
            - Pemeriksaan kehamilan rutin  
            - Minum air putih yang cukup  
            - Olahraga ringan sesuai anjuran  
            """)
        elif pred == 1:
            st.markdown("""
            - Pantau tekanan darah & gula darah  
            - Kurangi konsumsi gula & garam  
            - Istirahat cukup  
            - Kelola stres dengan baik  
            """)
        else:
            st.markdown("""
            - Segera konsultasi ke dokter atau bidan  
            - Pantau tekanan darah & gula darah secara intensif  
            - Hindari aktivitas berat  
            - Waspadai tanda bahaya kehamilan  
            """)

        # -----------------------------
        # DETAIL INPUT IBU
        # -----------------------------
        st.subheader("📊 Ringkasan Data Ibu")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usia", f"{age} tahun")
        col2.metric("Tekanan Darah", f"{sys}/{dia} mmHg")
        col3.metric("Suhu Tubuh", f"{temp_c} °C")

    st.markdown("---")
    st.caption(
        "⚠️ Hasil prediksi ini bersifat skrining awal dan "
        "**tidak menggantikan pemeriksaan dokter atau bidan**."
    )
