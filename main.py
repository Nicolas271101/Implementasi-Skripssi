import streamlit as st
import numpy as np
from process.preprocessor import preprocessing, vectorizer11, vectorizer12, vectorizer2, vectorizer31
from process.model import modelujicoba11, modelujicoba12, modelujicoba2, modelujicoba31, modelujicoba32, modelujicoba4

# Menggunakan CSS untuk styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 10px;
    }
    .title {
        color: #004d99;
        font-size: 2.5em;
        text-align: center;
    }
    <style>
    textarea {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }
    .result-positive {
        color: green;
        font-weight: bold;
    }
    .result-neutral {
        color: orange;
        font-weight: bold;
    }
    .result-negative {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    # .button {
    #     background-color: #004d99;
    #     color: white;
    #     padding: 10px 20px;
    #     border: none;
    #     border-radius: 5px;
    #     cursor: pointer;
    # }
    # .button:hover {
    #     background-color: #003366;
    # }

# Judul aplikasi
st.markdown('<h1 class="title">Analisis Sentimen Ulasan</h1>', unsafe_allow_html=True)


# Input teks
st.markdown('<div class="main">', unsafe_allow_html=True)
inp = st.text_area(label="inp", label_visibility="hidden", placeholder="Please input text...", height=250)

# Tombol submit dengan styling
# btn_submit_all = st.button("Submit All Models", class_="button")
btn_submit_all = st.button("Submit All Models")
# btn_submit_model11 = st.button("Submit Model 11", key='model11', help="Submit for Model 11", class_="button")
# btn_submit_model12 = st.button("Submit Model 12", key='model12', help="Submit for Model 12", class_="button")
# btn_submit_model2 = st.button("Submit Model 2", key='model2', help="Submit for Model 2", class_="button")
# btn_submit_model31 = st.button("Submit Model 31", key='model31', help="Submit for Model 31", class_="button")
# btn_submit_model32 = st.button("Submit Model 32", key='model32', help="Submit for Model 32", class_="button")
# btn_submit_model4 = st.button("Submit Model 4", key='model4', help="Submit for Model 4", class_="button")

if btn_submit_all:
    if inp == "":
        st.write("error")
    else:
        prep = preprocessing(inp)
        st.write(f"Preprocessed Input: {prep}")

        # Model 1.1
        transform = vectorizer11.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba11.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("VADER Lexicon Pertoken+ SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">VADER Lexicon Pertoken + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("VADER Lexicon Pertoken+ SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">VADER Lexicon Pertoken + SVM : Sentimen Netral</p>', unsafe_allow_html=True) 
        else:
            st.write("VADER Lexicon Pertoken+ SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">VADER Lexicon Pertoken + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)

        # Model 1.2
        transform = vectorizer12.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba12.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("VADER Lexicon Perulasan + SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">VADER Lexicon Perulasan + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("VADER Lexicon Perulasan + SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">VADER Lexicon Perulasan + SVM : Sentimen Netral</p>', unsafe_allow_html=True)
        else:
            st.write("VADER Lexicon Perulasan + SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">VADER Lexicon Perulasan + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)

        # Model 2
        transform = vectorizer2.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba2.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("Manual + SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">Manual + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("Manual + SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">Manual + SVM : Sentimen Netral</p>', unsafe_allow_html=True)
        else:
            st.write("Manual + SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">Manual + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)

        # Model 3.1
        transform = vectorizer31.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba31.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Netral</p>', unsafe_allow_html=True)
        else:
            st.write("VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">VADER Lexicon Pertoken + SMOTE + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)

        # Model 3.2
        transform = vectorizer31.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba32.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Netral</p>', unsafe_allow_html=True)
        else:
            st.write("VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">VADER Lexicon Perulasan + SMOTE + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)

        # Model 4
        transform = vectorizer2.transform([" ".join(prep)])
        # trans = loaded_vec.fit_transform([transform])
        label = modelujicoba4.predict(np.asarray(transform.todense()))
        if label[0] == 1:
            st.write("Manual + SMOTE + SVM : Sentimen Positif")
            st.markdown('<p class="result-positive">Manual + SMOTE + SVM : Sentimen Positif</p>', unsafe_allow_html=True)
        elif label[0] == 0:
            st.write("Manual + SMOTE + SVM : Sentimen Netral")
            st.markdown('<p class="result-neutral">Manual + SMOTE + SVM : Sentimen Netral</p>', unsafe_allow_html=True)
        else:
            st.write("Manual + SMOTE + SVM : Sentimen Negatif")
            st.markdown('<p class="result-negative">Manual + SMOTE + SVM : Sentimen Negatif</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)