import streamlit as st
import re
import unicodedata
import joblib
import torch

from langdetect import detect, DetectorFactory, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="📱",
    layout="centered"
)


# ============================================================
# LOAD SENTIMENT MODEL AND TF-IDF
# ============================================================

@st.cache_resource
def load_sentiment_model():

    model = joblib.load("tuned_svm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    return model, tfidf


model, tfidf = load_sentiment_model()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

DetectorFactory.seed = 42


def detect_language(text):

    text = str(text).strip()

    if not text:
        return "unknown"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"


# ============================================================
# LANGUAGE MAPPING FOR NLLB
# ============================================================

language_map = {

    "en": "eng_Latn",

    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "ne": "npi_Deva",

    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",

    "ru": "rus_Cyrl",
    "ar": "arb_Arab",

    "ja": "jpn_Jpan",
    "ko": "kor_Hang",

    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant"
}


# ============================================================
# LOAD NLLB TRANSLATION MODEL
# ============================================================
# IMPORTANT:
# This function is NOT called when the app starts.
#
# It is called only when a non-English supported language
# actually needs translation.
#
# @st.cache_resource prevents unnecessary reloading during
# Streamlit reruns.
# ============================================================

@st.cache_resource
def load_translation_model():

    st.info(
        "Loading translation model for the first multilingual prediction..."
    )

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    translation_model = translation_model.to(device)

    return tokenizer, translation_model, device


# ============================================================
# TRANSLATE REVIEW TO ENGLISH
# ============================================================

def translate_to_english(text):

    text = str(text).strip()

    if not text:
        return text, "unknown"

    # Detect language
    detected_lang = detect_language(text)

    # --------------------------------------------------------
    # If already English, no translation is required
    # --------------------------------------------------------

    if detected_lang == "en":

        return text, detected_lang


    # --------------------------------------------------------
    # Check whether language is supported
    # --------------------------------------------------------

    if detected_lang not in language_map:

        return text, detected_lang


    # --------------------------------------------------------
    # Load NLLB ONLY NOW
    # --------------------------------------------------------

    tokenizer, translation_model, device = load_translation_model()


    # Source language required by NLLB
    source_language = language_map[detected_lang]

    try:

        tokenizer.src_lang = source_language

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Move input tensors to CPU/GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Generate English translation
        # ----------------------------------------------------

        translated_tokens = translation_model.generate(

            **inputs,

            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "eng_Latn"
            ),

            max_length=512
        )


        # Convert generated tokens back to text
        translated_text = tokenizer.batch_decode(

            translated_tokens,

            skip_special_tokens=True

        )[0]


        return translated_text, detected_lang


    except Exception as e:

        st.warning(
            f"Translation could not be completed: {e}"
        )

        # If translation fails, return original text
        return text, detected_lang


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove punctuation while preserving Unicode letters
    text = "".join(

        char

        for char in text

        if not unicodedata.category(char).startswith("P")

    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title(
    "📱 Customer Review Sentiment Analysis"
)


st.write(
    "Enter a customer review to predict whether the "
    "sentiment is Positive, Negative, or Neutral."
)


st.info(
    "The application supports English and selected "
    "multilingual reviews. Non-English reviews are "
    "automatically translated into English before "
    "sentiment prediction."
)


# ============================================================
# USER INPUT
# ============================================================

review = st.text_area(

    "Enter Customer Review",

    placeholder=(
        "Example: The phone has an excellent camera "
        "and battery life."
    ),

    height=150
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("🔍 Predict Sentiment"):


    # --------------------------------------------------------
    # Check empty input
    # --------------------------------------------------------

    if not review.strip():

        st.warning(
            "Please enter a review."
        )


    else:

        # ====================================================
        # STEP 1: DETECT LANGUAGE
        # ====================================================

        detected_language = detect_language(review)


        # ====================================================
        # STEP 2: TRANSLATE IF REQUIRED
        # ====================================================

        translated_text, detected_language = translate_to_english(
            review
        )


        # ====================================================
        # STEP 3: CLEAN TEXT
        # ====================================================

        cleaned_review = clean_text(
            translated_text
        )


        # ====================================================
        # STEP 4: TF-IDF TRANSFORMATION
        # ====================================================
        # IMPORTANT:
        # Use transform(), NOT fit_transform()
        # because the TF-IDF vectorizer was already fitted
        # during model training.
        # ====================================================

        review_vector = tfidf.transform(
            [cleaned_review]
        )


        # ====================================================
        # STEP 5: SENTIMENT PREDICTION
        # ====================================================

        prediction = model.predict(
            review_vector
        )[0]


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        st.subheader(
            "Prediction"
        )


        if prediction == "Positive":

            st.success(
                "😊 Positive Sentiment"
            )


        elif prediction == "Negative":

            st.error(
                "😞 Negative Sentiment"
            )


        else:

            st.warning(
                "😐 Neutral Sentiment"
            )


        # ====================================================
        # DISPLAY LANGUAGE
        # ====================================================

        st.write(
            "**Detected Language:**",
            detected_language
        )


        # ====================================================
        # DISPLAY TRANSLATION
        # ====================================================

        if detected_language != "en":

            st.write(
                "**Translated Review:**"
            )

            st.write(
                translated_text
            )


        # ====================================================
        # DISPLAY PROCESSED TEXT
        # ====================================================

        st.write(
            "**Processed Review:**"
        )

        st.write(
            cleaned_review
        )