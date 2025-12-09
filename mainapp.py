import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Stunting & Risiko Ibu", layout="centered")
st.title("Sistem Prediksi Stunting Anak & Risiko Kesehatan Ibu")

st.markdown("""
Aplikasi ini menyediakan dua analisis:
1️⃣ Prediksi Stunting Balita  
2️⃣ Prediksi Risiko Kesehatan Ibu Hamil
""")

anak_dt = joblib.load("anak_decision_tree.pkl")
anak_rf = joblib.load("anak_random_forest_compressed.pkl")
anak_scaler = joblib.load("anak_scaler.pkl")

ibu_dt = joblib.load("ibu_decision_tree.pkl")
ibu_rf = joblib.load("ibu_random_forest.pkl")
ibu_scaler = joblib.load("ibu_scaler.pkl")

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["👶🏻 Prediksi Stunting Anak", "🤰🏻 Prediksi Risiko Kesehatan Ibu"]
)

if menu == "👶🏻 Prediksi Stunting Anak":
    st.header("Prediksi Stunting Anak 👶🏻")

    gender_map = {"Laki-laki": 0, "Perempuan": 1}

    gender = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
    age = st.number_input("Usia Anak (bulan)", 0, 60, 12)
    birth_weight = st.number_input("Berat Lahir (kg)", 0.5, 5.0, 3.0)
    birth_length = st.number_input("Panjang Lahir (cm)", 30.0, 60.0, 49.0)
    body_weight = st.number_input("Berat Badan Saat Ini (kg)", 1.0, 20.0, 10.0)
    body_length = st.number_input("Tinggi Badan Saat Ini (cm)", 40.0, 120.0, 70.0)
    breastfeeding = st.selectbox("ASI Eksklusif?", ["Ya", "Tidak"])

    bf = 1 if breastfeeding == "Ya" else 0
    g = gender_map[gender]

    data = np.array([[g, age, birth_weight, birth_length, body_weight, body_length, bf]])
    data_scaled = anak_scaler.transform(data)

    model_choice = st.radio("Model:", ["Decision Tree", "Random Forest"])
    model = anak_dt if model_choice == "Decision Tree" else anak_rf

    if st.button("Prediksi Stunting"):
        pred = model.predict(data_scaled)[0]
        akurasi = model.score(anak_scaler.transform(data), model.predict(data))
        st.metric("Akurasi Model", f"{akurasi*100:.2f}%")

        if pred == 1:
            st.error("⚠️ Anak terindikasi Stunting.")
        else:
            st.success("✅ Anak tidak stunting.")

        st.subheader("Saran Kesehatan Anak")

        if pred == 1:
            st.markdown("""
            - Tingkatkan asupan protein (telur, ikan, ayam, tempe, tahu)  
            - Tambahkan buah & sayuran  
            - Periksa pertumbuhan rutin di posyandu  
            - Perbaiki sanitasi lingkungan  
            - Konsultasi dokter bila pertumbuhan stagnan  
            """)
        else:
            st.markdown("""
            - Jaga pola makan seimbang  
            - Batasi makanan manis dan instan  
            - Kontrol rutin di posyandu  
            - Berikan stimulasi perkembangan  
            - Pastikan tidur cukup  
            """)

elif menu == "🤰🏻 Prediksi Risiko Kesehatan Ibu":
    st.header("Prediksi Risiko Kesehatan Ibu 🤰🏻")

    age = st.number_input("Usia Ibu", 15, 50, 28)
    sys = st.number_input("Systolic BP", 80, 200, 120)
    dia = st.number_input("Diastolic BP", 50, 130, 80)
    bs = st.number_input("Blood Sugar", 1.0, 30.0, 7.0)
    temp = st.number_input("Temperatur Tubuh (°F)", 90.0, 110.0, 98.0)
    heart = st.number_input("Heart Rate", 50, 200, 100)

    data = np.array([[age, sys, dia, bs, temp, heart]])
    data_scaled = ibu_scaler.transform(data)

    model_choice = st.radio("Model:", ["Decision Tree", "Random Forest"])
    model = ibu_dt if model_choice == "Decision Tree" else ibu_rf

    if st.button("Prediksi Risiko Ibu"):
        pred = model.predict(data_scaled)[0]
        akurasi = model.score(ibu_scaler.transform(data), model.predict(data))
        st.metric("Akurasi Model", f"{akurasi*100:.2f}%")

        if pred == 0:
            st.success("Risiko Rendah 🟢")
        elif pred == 1:
            st.warning("Risiko Sedang 🟡")
        else:
            st.error("Risiko Tinggi 🔴")

        st.subheader("Saran Kesehatan Ibu")

        if pred == 0:
            st.markdown("""
            - Makan seimbang  
            - Minum cukup  
            - Cek kehamilan rutin  
            - Olahraga ringan  
            """)
        elif pred == 1:
            st.markdown("""
            - Pantau tekanan darah  
            - Kurangi konsumsi gula  
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

st.sidebar.markdown("---")
st.sidebar.caption("Developed with 💖 using Streamlit")
