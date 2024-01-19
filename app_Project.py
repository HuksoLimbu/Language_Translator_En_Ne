import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModel

# Load your trained Keras model
model = tf.keras.models.load_model('my_model.h5')

# Streamlit app layout 
st.title("Language Translator English to Nepali")
input_text = st.text_area("Enter text to translate:", " ")

# Create a tokenizer from Hugging Face transformers library
# tokenizer = AutoTokenizer.from_pretrained("my_model.h5")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Define the Streamlit app
def main():
    # st.title('Language Translation with Transformer Model')

    # Text input for translation
    text_input = st.text_input("Enter text in source language:", value="")

    if text_input:
        # Tokenize the input text
        inputs = tokenizer(text_input, return_tensors="tf", padding=True, truncation=True)

        # Generate translation
        translation = model.generate(**inputs)

        # Decode the generated translation
        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
        
        # Display translation result
        st.write(f"Translated text in target language: {translated_text}")

if __name__ == '__main__':
    main()


