import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #00FFAA;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #BBBBBB;
        margin-bottom: 30px;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #00FFAA;
        font-size: 18px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        background-color: #00FFAA;
        color: black;
        font-size: 20px;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #00CC88;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load Dataset
data = pd.read_csv("spam_ham_dataset.csv", encoding='latin-1')

# Keep needed columns
data = data[['label', 'text']]
data.columns = ['label', 'message']

# Encode labels
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label_num'],
    test_size=0.2,
    random_state=42
)
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# UI
st.markdown('<div class="title">📧 Spam Email Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered email and SMS spam detection system</div>', unsafe_allow_html=True)

# Accuracy Card
st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

# User Input
user_message = st.text_area(
    "Enter your message below:",
    height=180,
    placeholder="Type your email or SMS here..."
)

# Predict Button
if st.button("🔍 Analyze Message"):

    if user_message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        transformed_message = vectorizer.transform([user_message])
        prediction = model.predict(transformed_message)[0]

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
            st.write("This message appears suspicious or promotional.")

        else:
            st.success("✅ SAFE MESSAGE (HAM)")
            st.write("This message looks safe and legitimate.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, TF-IDF, and Naive Bayes Machine Learning")