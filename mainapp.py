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
    "1️⃣ Prediksi Stunting Pada Anak  \n"
    "2️⃣ Prediksi Risiko Kesehatan Ibu"
)

anak_model = joblib.load("bayi_random_forest.pkl")
anak_scaler = joblib.load("bayi_scaler.pkl")
AKURASI_ANAK = 0.865

ibu_model = joblib.load("ibu_random_forest.pkl")
ibu_scaler = joblib.load("ibu_scaler.pkl")
AKURASI_IBU = 0.8473

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["👶🏻 Prediksi Stunting Anak", "🤰🏻 Prediksi Risiko Kesehatan Ibu"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed with 💖 using Streamlit")

def is_definitely_stunted(age, height):
    if age >= 36:
        return height < 90
    elif age >= 24:
        return height < 85
    elif age >= 12:
        return height < 75
    else:
        return False

def is_growth_normal(age, height, weight):
    if age < 12:
        return height >= 70 and weight >= 7
    elif age < 24:
        return height >= 80 and weight >= 9
    elif age < 36:
        return height >= 85 and weight >= 11
    else:
        return height >= 95 and weight >= 12

if menu == "👶🏻 Prediksi Stunting Anak":

    st.header("👶🏻 Prediksi Stunting Anak")

    st.markdown("""
    Prediksi dilakukan menggunakan pendekatan **hybrid**  
    (aturan pertumbuhan berbasis usia + machine learning).
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

    if st.button("🔍 Prediksi Stunting Anak"):

        data = np.array([[g, age, birth_weight, birth_length, body_weight, body_length, bf]])
        data_scaled = anak_scaler.transform(data)
        pred_model = anak_model.predict(data_scaled)[0]

        if is_definitely_stunted(age, body_length):
            final_pred = 1
            decision_source = "Rule-based (stunting jelas berdasarkan usia & tinggi)"
        elif is_growth_normal(age, body_length, body_weight):
            final_pred = 0
            decision_source = "Rule-based (pertumbuhan sesuai usia)"
        else:
            final_pred = pred_model
            decision_source = "Prediksi model machine learning"

        st.metric("Akurasi Model (Validasi)", f"{AKURASI_ANAK*100:.2f}%")

        if final_pred == 1:
            st.error("⚠️ Anak **terindikasi stunting**.")
        else:
            st.success("✅ Anak **tidak stunting**.")

        st.caption(f"Sumber keputusan: **{decision_source}**")

        st.info(
            "Penilaian stunting difokuskan pada hubungan antara usia dan tinggi badan. "
            "Aturan digunakan untuk mencegah kesalahan prediksi pada kasus ekstrem."
        )

        st.subheader("📊 Detail Input Anak")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usia", f"{age} bulan")
        col2.metric("Berat Badan", f"{body_weight} kg")
        col3.metric("Tinggi Badan", f"{body_length} cm")

    st.markdown("---")
    st.caption("⚠️ Aplikasi ini hanya alat bantu skrining awal dan **tidak menggantikan diagnosis dokter**.")

elif menu == "🤰🏻 Prediksi Risiko Kesehatan Ibu":

    st.header("🤰🏻 Prediksi Risiko Kesehatan Ibu")

    st.markdown("""
    Prediksi risiko kesehatan ibu dilakukan berdasarkan parameter klinis
    dan hasil model machine learning.
    """)

    age = st.number_input("Usia Ibu (tahun)", 15, 50, 28)
    sys = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
    dia = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 80)
    bs = st.number_input("Blood Sugar", 1.0, 30.0, 7.0)

    temp_c = st.number_input("Temperatur Tubuh (°C)", 30.0, 45.0, 36.5)
    temp_f = (temp_c * 9 / 5) + 32

    heart = st.number_input("Heart Rate (bpm)", 50, 200, 100)

    if st.button("🔍 Prediksi Risiko Ibu"):

        data = np.array([[age, sys, dia, bs, temp_f, heart]])
        data_scaled = ibu_scaler.transform(data)
        pred_model = ibu_model.predict(data_scaled)[0]

        if pred_model == 2 and sys < 140 and dia <= 90 and bs < 8 and heart < 100:
            final_pred = 1
            decision_note = "Disesuaikan (parameter mendekati batas risiko tinggi)"
        else:
            final_pred = pred_model
            decision_note = "Prediksi model machine learning"

        st.metric("Akurasi Model (Validasi)", f"{AKURASI_IBU*100:.2f}%")

        if final_pred == 0:
            st.success("🟢 Risiko Rendah")
        elif final_pred == 1:
            st.warning("🟡 Risiko Sedang")
        else:
            st.error("🔴 Risiko Tinggi")

        st.caption(f"Sumber keputusan: **{decision_note}**")

        st.subheader("📊 Ringkasan Data Ibu")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usia", f"{age} tahun")
        col2.metric("Tekanan Darah", f"{sys}/{dia} mmHg")
        col3.metric("Suhu Tubuh", f"{temp_c} °C")

    st.markdown("---")
    st.caption("⚠️ Hasil prediksi ini bersifat skrining awal dan **tidak menggantikan pemeriksaan dokter atau bidan**.")
