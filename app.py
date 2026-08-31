import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("AI Translation")

model_name = "facebook/nllb-200-distilled-600M"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

language_codes = {
    "English": "eng_Latn",
    "Indonesian": "ind_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang"
}

# Two columns
col1, col2 = st.columns(2)


# Source
with col1:
    source_language = st.selectbox(
        "Source language",
        list(language_codes.keys())
    )

# Target
with col2:
    target_language = st.selectbox(
        "Target language",
        list(language_codes.keys())
    )

    result_placeholder = st.empty()


text = st.text_area("Enter text")

# Translate
if st.button("Translate"):

    if text.strip():

        source_code = language_codes[source_language]
        target_code = language_codes[target_language]

        # Set source language
        tokenizer.src_lang = source_code

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt"
        )

        # Generate translation
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code)
        )

        # Convert tokens back to text
        result = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        st.subheader("Translation")

        st.write(result[0])

    else:
        st.warning("Please enter some text.")