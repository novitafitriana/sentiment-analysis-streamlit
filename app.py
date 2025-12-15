import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Sentimen Pegadaian", layout="wide")

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("data/hasil_sentimen.csv")

# ============================
# SIDEBAR NAVIGASI
# ============================
st.sidebar.title("📘 ANALISIS SENTIMEN")
menu = st.sidebar.radio(
    "Menu Utama",
    ["📁 Data Understanding", 
     "⚙️ Data Preparation", 
     "📊 Modeling & Evaluation", 
     "📝 Testing"]
)

# ===============================================================
# 1️⃣ DATA UNDERSTANDING (Updated)
# ===============================================================
if menu == "📁 Data Understanding":
    st.title("📁 Data Understanding")

    st.subheader("🔍 Preview Dataset")
    st.dataframe(df.head())

    st.write(f"📌 Total Data: **{len(df)}** ulasan")

    # ============================
    #  DISTRIBUSI RATING
    # ============================
    st.subheader("⭐ Distribusi Rating Pengguna")

    fig_rating = px.histogram(df, x="score", nbins=5, color="score",
                              title="Distribusi Rating (1–5)")
    st.plotly_chart(fig_rating, use_container_width=True)

    # ============================
    # DISTRIBUSI PANJANG REVIEW
    # ============================
    st.subheader("✍ Distribusi Panjang Ulasan (jumlah kata)")

    fig_len = px.histogram(df, x="review_length", nbins=30,
                           title="Distribusi Panjang Kalimat Ulasan")
    st.plotly_chart(fig_len, use_container_width=True)

    # ============================
    # TOP FREQUENT WORDS
    # ============================
    st.subheader("📌 Kata yang Paling Sering Muncul")

    text = " ".join(df["content"].astype(str)).lower().split()
    top_words = Counter(text).most_common(15)

    top_words_df = pd.DataFrame(top_words, columns=["Word", "Count"])
    fig_topwords = px.bar(top_words_df, x="Word", y="Count", title="Top 15 Kata Terbanyak")
    st.plotly_chart(fig_topwords, use_container_width=True)

    # ============================
    # WORDCLOUD ASLI
    # ============================
    st.subheader("☁️ WordCloud Sebelum Preprocessing")

    wc_awal = WordCloud(width=1000, height=400, background_color="white").generate(" ".join(df["content"]))
    fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
    ax_wc.imshow(wc_awal)
    ax_wc.axis("off")
    st.pyplot(fig_wc)
# ===============================================================
# 2️⃣ DATA PREPARATION
# ===============================================================
if menu == "⚙️ Data Preparation":
    st.title("⚙️ Data Preparation")

    st.subheader("📌 Kolom Hasil Preprocessing")
    kolom_dp = ["content", "clean_review", "normalized_content", "tokens", "stemmed_text"]
    st.dataframe(df[kolom_dp].head())

    st.subheader("☁️ WordCloud Setelah Preprocessing")
    text_after = " ".join(df["stemmed_text"].astype(str))
    wc_after = WordCloud(width=900, height=400, background_color="white").generate(text_after)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.imshow(wc_after, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

# ===============================================================
# 3️⃣ MODELING & EVALUATION
# ===============================================================
if menu == "📊 Modeling & Evaluation":
    st.title("📊 Modeling & Evaluation")

    st.subheader("Pilih Label yang Ingin Ditampilkan")
    opsi = st.selectbox(
        "Pilih salah satu:",
        ["label", "label_lexicon", "sentiment_indobert"]
    )

    st.write(f"### 🔍 Menampilkan hasil untuk: **{opsi}**")

    col1, col2 = st.columns(2)

    # =======================
    # KIRI = HISTOGRAM SENTIMEN
    # =======================
    with col1:
        st.subheader("📊 Distribusi Sentimen")
        fig = px.histogram(df, x=opsi, color=opsi)
        st.plotly_chart(fig, use_container_width=True)

    # =======================
    # KANAN = WORDCLOUD PER LABEL
    # =======================
    with col2:
        st.subheader("☁️ WordCloud Berdasarkan Label")
        text_wc = " ".join(df[df[opsi] == df[opsi].mode()[0]]["stemmed_text"].astype(str))

        wc_label = WordCloud(width=900, height=400, background_color="white").generate(text_wc)

        fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
        ax_wc.imshow(wc_label, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # =======================
    # BAWAH = DATASET
    # =======================
    st.subheader("📑 Dataset Hasil Sentimen")
    st.dataframe(df[[opsi, "content", "stemmed_text"]].head())


# ===============================================================
# 4️⃣ TESTING (PREDIKSI MANUAL)
# ===============================================================
if menu == "📝 Testing":
    from transformers import pipeline

    st.title("📝 Testing Sentimen IndoBERT")

    user_text = st.text_area("Masukkan teks ulasan:")

    if st.button("Prediksi"):
        model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        pipe = pipeline("text-classification", model=model_name, tokenizer=model_name)

        if user_text.strip() == "":
            st.warning("Masukkan teks dulu ya.")
        else:
            hasil = pipe(user_text)[0]["label"]
            st.success(f"Hasil Prediksi: **{hasil}**")

