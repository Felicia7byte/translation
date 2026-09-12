# translation
An AI-powered text translation application built with Python and Streamlit using the Meta NLLB-200 multilingual translation model from Hugging Face Transformers.
# Features
- Translate text between multiple languages
- Support English, Indonesian, Chinese, Japanese, and Korean
- Select source and target languages independently
- Simple and interactive interface built with Streamlit
- Display the translated text directly in the browser
# Tech Stack
Python, Streamlit, Hugging Face Transformers, NLLB-200 (facebook/nllb-200-distilled-600M)
# How It Works
Enter Text → Select Source Language → Select Target Language → Tokenization → NLLB-200 Model Inference → Decode Output → Translated Text

The application uses the pre-trained facebook/nllb-200-distilled-600M model to translate text between supported languages. The selected source language is provided to the tokenizer, while the target language is specified during model generation.
# Model
Model: NLLB-200 Distilled 600M
# Live Demo
https://ai-translation-test.streamlit.app/
