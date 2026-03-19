import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#tytul w karcie i ikona
st.set_page_config(page_title="Klasyfikator Galaktyk", page_icon="🌌")

st.title("Rozpoznawanie typu galaktyki za pomocą sieci neuronowej")
st.write("Wgraj zdjęcie galaktyki, a sieć neuronowa (CNN) oceni jej kształt")


#ladowanie modelu z dysku
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('galaxy_classifier.keras')

try:
    model = load_model()
    st.success("Model załadowany poprawnie, czekam na zdjęcie.")
except Exception as e:
    st.error(f"Błąd ładowania modelu Upewnij się, że plik 'galaxy_classifier.keras' jest w tym samym folderze co 'app.py'. Szczegóły: {e}")
    st.stop() #jesli nie ma modelu to zatrzymuje dzialanie apki

#3 przycisk do wgrywania plikow z urzadzenia
uploaded_file = st.file_uploader("Wybierz zdjęcie galaktyki (JPG/PNG)...", type=["JPG", "JPEG", "PNG"])

if uploaded_file is not None:
    #wyswietlanie wgranego zdjecia na ekranie
    image = Image.open(uploaded_file)
    st.image(Image, caption='Twoja galaktyka do analizy', use_column_width=True)

    st.write("Analiza zdjęcia galaktyki...")

    #4 przygotowanie zdjecia pod model
    img = image.convert('RGB')
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 #normalizacja

    #predykcja

    wynik = model.predict(img_array)


    prob_el = wynik[0][0] * 100
    prob_sp = wynik[0][1] * 100

    #wyswietlenie wynikow
    st.subheader("Wyniki analizy:")
    st.write(f"Prawdopodobieństwo, że to galaktyka eliptczna: {prob_el:.2f}%")
    st.write(f"Prawdopodobieństwo, że to galaktyka spiralna: {prob_sp:.2f}%")

    if prob_sp > prob_el:
        st.success("Werdykt: Galaktyka Spiralna!")
    else: 
        st.info("Werdykt: Galaktyka Eliptyczna!")





    