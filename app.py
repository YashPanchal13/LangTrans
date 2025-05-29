# app.py

import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ✅ Set page config FIRST
st.set_page_config(page_title="🌍 Multilingual Translator", layout="centered")

# Load M2M100 model
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Language code map
lang_code = {
    "English": "en",
    "Chinese": "zh",
    "Arabic": "ar"
}

# Translate function
def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = lang_code[source_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(lang_code[target_lang])
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Streamlit UI
st.title("🌍 Multilingual Translator")
st.markdown("Translate between **English**, **Chinese**, and **Arabic** using Facebook's M2M100 model.")

text = st.text_area("Enter Text:", height=150)
src_lang = st.selectbox("Source Language", list(lang_code.keys()))
tgt_lang = st.selectbox("Target Language", list(lang_code.keys()))

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text.")
    elif src_lang == tgt_lang:
        st.info("Source and target languages are the same.")
        st.text(text)
    else:
        result = translate_text(text, src_lang, tgt_lang)
        st.success("Translated Text:")
        st.markdown(f"**{result}**")


#-------------------------------------------------------------------------------------------- FLASK
# from flask import Flask, request, jsonify
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# app = Flask(__name__)

# model_name = "facebook/m2m100_418M"
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# lang_code = {"English": "en", "Chinese": "zh", "Arabic": "ar"}

# def translate_text(text, source_lang, target_lang):
#     tokenizer.src_lang = lang_code[source_lang]
#     encoded = tokenizer(text, return_tensors="pt")
#     generated_tokens = model.generate(
#         **encoded,
#         forced_bos_token_id=tokenizer.get_lang_id(lang_code[target_lang])
#     )
#     return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# @app.route("/translate", methods=["POST"])
# def translate():
#     data = request.json
#     text = data["text"]
#     src = data["source_lang"]
#     tgt = data["target_lang"]
#     translated = translate_text(text, src, tgt)
#     return jsonify({"translated_text": translated})

# if __name__ == "__main__":
#     app.run(debug=True)
