import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler

# Ensure necessary NLTK data is downloaded for tokenization, stopwords, and lemmatization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the saved models and necessary components
lr_model = joblib.load('lr_model.pkl')  # Logistic Regression model
svc_model = joblib.load('svm_model.pkl')  # Support Vector Machine model
label_encoder = joblib.load('label_encoder.pkl')  # Load the saved LabelEncoder
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the saved TF-IDF vectorizer
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

def clean_text(text):
    """
    Cleans the input text by removing special characters, digits, and HTML tags.
    
    Parameters:
    - text: The raw input text to be cleaned.
    
    Returns:
    - A cleaned version of the input text (lowercased and without non-alphabetic characters).
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters and digits
    return text

def preprocess_text(text):
    """
    Tokenizes the input text, removes stopwords, and lemmatizes the words.
    
    Parameters:
    - text: The input text to be preprocessed.
    
    Returns:
    - The preprocessed text as a string (tokens lemmatized and stopwords removed).
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens (convert words to their base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove extra spaces and rejoin the tokens into a string
    text = ' '.join(tokens)
    text = ' '.join(text.split())  # Remove extra spaces between words
    return text

def predict_sentiment(model, label_encoder, user_input):
    """
    Preprocesses the input text, transforms it using TF-IDF and scaler, and predicts the sentiment.
    
    Parameters:
    - model: The trained model to predict sentiment (either Logistic Regression or SVM).
    - label_encoder: The LabelEncoder to map numeric predictions back to sentiment labels.
    - user_input: The raw review text from the user.
    
    Returns:
    - The predicted sentiment label (Positive, Negative, or Neutral).
    """
    # Clean and preprocess the user input text
    cleaned_text = clean_text(user_input)
    preprocessed_text = preprocess_text(cleaned_text)

    # Apply TF-IDF transformation to the preprocessed text
    tfidf_features = tfidf_vectorizer.transform([preprocessed_text])

    # Apply scaling to the transformed TF-IDF features
    scaled_features = scaler.transform(tfidf_features)

    # Predict sentiment using the selected model
    numerical_prediction = model.predict(scaled_features)

    # Map the numerical prediction back to the original sentiment labels using the label encoder
    sentiment = label_encoder.inverse_transform(numerical_prediction)

    return sentiment[0]

# Streamlit UI setup
st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis of Amazon Reviews")
st.markdown("---")

# User input section: A text box for the user to enter their review
user_input = st.text_area("Enter your review text:")

# Option to choose between Logistic Regression and Support Vector Machine models
model_option = st.radio(
    "**Choose a Model for Sentiment Analysis:**",
    ('Logistic Regression: Fast results with reasonable accuracy.',
     'Support Vector Machine: Accurate but more time-consuming.')
)

# Map the user-friendly model choice to the corresponding model
if model_option == 'Logistic Regression: Fast results with reasonable accuracy.':
    selected_model = lr_model  # Logistic Regression (faster model)
else:
    selected_model = svc_model  # Support Vector Machine (more accurate, slower model)

# Predict sentiment when the user clicks the "Predict Sentiment" button
if st.button("Predict Sentiment"):
    if user_input.strip():
        with st.spinner("Processing..."):
            sentiment = predict_sentiment(selected_model, label_encoder, user_input)
            st.subheader(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to predict the sentiment.")
