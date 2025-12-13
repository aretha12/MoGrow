import streamlit as st
import numpy as np
import joblib

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

anak_model = joblib.load("bayi_random_forest.pkl")
anak_scaler = joblib.load("bayi_scaler.pkl")
AKURASI_ANAK = 0.8712

ibu_model = joblib.load("ibu_random_forest.pkl")
ibu_scaler = joblib.load("ibu_scaler.pkl")
AKURASI_IBU = 0.8374

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["👶🏻 Prediksi Stunting Anak", "🤰🏻 Prediksi Risiko Kesehatan Ibu"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed with 💖 using Streamlit")

if menu == "👶🏻 Prediksi Stunting Anak":

    st.header("👶🏻 Prediksi Stunting Anak")

    st.markdown("""
    Prediksi dilakukan menggunakan **machine learning (Random Forest)**  
    dan **aturan pertumbuhan (rule-based override)** untuk menghindari
    kesalahan prediksi pada anak dengan pertumbuhan normal.
    """)

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

    if st.button("🔍 Prediksi Stunting"):

        data = np.array([[g, age, birth_weight, birth_length, body_weight, body_length, bf]])
        data_scaled = anak_scaler.transform(data)

        pred_model = anak_model.predict(data_scaled)[0]

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
        st.metric("Akurasi Model (Validasi)", f"{AKURASI_ANAK*100:.2f}%")

        if final_pred == 1:
            st.error("⚠️ Anak **terindikasi stunting**.")
        else:
            st.success("✅ Anak **tidak stunting**.")

        st.caption(f"Sumber keputusan: **{decision_source}**")
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
        st.subheader("📊 Detail Input")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usia", f"{age} bulan")
        col2.metric("Berat Badan", f"{body_weight} kg")
        col3.metric("Tinggi Badan", f"{body_length} cm")

    st.markdown("---")
    st.caption("⚠️ Aplikasi ini hanya alat bantu skrining awal dan **tidak menggantikan diagnosis dokter**.")

elif menu == "🤰🏻 Prediksi Risiko Kesehatan Ibu":

    st.header("🤰🏻 Prediksi Risiko Kesehatan Ibu")

    age = st.number_input("Usia Ibu", 15, 50, 28)
    sys = st.number_input("Systolic BP", 80, 200, 120)
    dia = st.number_input("Diastolic BP", 50, 130, 80)
    bs = st.number_input("Blood Sugar", 1.0, 30.0, 7.0)
    temp = st.number_input("Temperatur Tubuh (°F)", 90.0, 110.0, 98.0)
    heart = st.number_input("Heart Rate", 50, 200, 100)

    if st.button("🔍 Prediksi Risiko Ibu"):

        data = np.array([[age, sys, dia, bs, temp, heart]])
        data_scaled = ibu_scaler.transform(data)
        pred = ibu_model.predict(data_scaled)[0]

        st.metric("Akurasi Model (Validasi)", f"{AKURASI_IBU*100:.2f}%")

        if pred == 0:
            st.success("🟢 Risiko Rendah")
        elif pred == 1:
            st.warning("🟡 Risiko Sedang")
        else:
            st.error("🔴 Risiko Tinggi")

        st.subheader("🩺 Saran Kesehatan Ibu")

        if pred == 0:
            st.markdown("""
            - Konsumsi makanan seimbang  
            - Periksa kehamilan rutin  
            - Minum air cukup  
            - Olahraga ringan  
            """)
        elif pred == 1:
            st.markdown("""
            - Pantau tekanan darah  
            - Kurangi makanan tinggi gula  
            - Istirahat cukup  
            - Hindari stres  
            """)
        else:
            st.markdown("""
            - Segera konsultasi ke dokter  
            - Pantau tekanan darah & gula darah  
            - Hindari aktivitas berat  
            - Waspadai gejala bahaya  
            """)
