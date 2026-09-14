import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# PAGE CONFIG
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#00FFAA;
}

.subtitle{
    text-align:center;
    color:#BBBBBB;
    font-size:18px;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

#TEXT CLEANING
def clean_text(text):

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

#TRAIN MODEL
@st.cache_resource
def train_model():

    df = pd.read_csv("spam_ham_dataset.csv", encoding="latin-1")

    df = df[['label', 'text']]

    df['text'] = df['text'].apply(clean_text)

    df['label_num'] = df['label'].map({
        'ham': 0,
        'spam': 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label_num'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_num']
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return (
        model,
        vectorizer,
        accuracy,
        precision,
        recall,
        f1
    )

(
    model,
    vectorizer,
    accuracy,
    precision,
    recall,
    f1
) = train_model()

# HEADER
st.markdown(
    '<div class="title">📧 Spam Email Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by TF-IDF + Linear SVM</div>',
    unsafe_allow_html=True
)

# METRICS
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

with st.expander("View Model Metrics"):

    st.write(
        f"Precision: {precision:.3f}"
    )

    st.write(
        f"Recall: {recall:.3f}"
    )
# INPUT
user_message = st.text_area(
    "Enter Email or SMS Message",
    height=180,
    placeholder="Type your email here..."
)
# PREDICTION
if st.button("🔍 Analyze a Message"):

    if user_message.strip() == "":

        st.warning(
            "Please enter a message."
        )

    else:

        cleaned_message = clean_text(
            user_message
        )

        transformed = vectorizer.transform(
            [cleaned_message]
        )

        prediction = model.predict(
            transformed
        )[0]

        score = abs(
            model.decision_function(
                transformed
            )[0]
        )

        st.markdown("---")

        if prediction == 1:

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )

            st.write(
                "This email appears suspicious."
            )

        else:

            st.success(
                "✅ SAFE MESSAGE"
            )

            st.write(
                "This email appears legitimate."
            )

        st.info(
            f"Confidence Score: {score:.2f}"
        )

        word_count = len(
            user_message.split()
        )

        char_count = len(
            user_message
        )

        st.write(
            f"Words: {word_count}"
        )

        st.write(
            f"Characters: {char_count}"
        )

# FOOTER
st.markdown("---")

st.caption(
    "Built using Streamlit, TF-IDF and Linear SVM"
)
