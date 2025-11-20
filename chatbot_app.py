
import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('preprocessing.pkl', 'rb') as f:
        preprocessing = pickle.load(f)
    return model, vectorizer, preprocessing

model, vectorizer, preprocessing = load_model()
lemmatizer = preprocessing['lemmatizer']
stop_words = preprocessing['stop_words']
classes = preprocessing['classes']

# Preprocessing function
def advanced_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(cleaned_tokens)

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Status Chatbot")
st.markdown("### Detect mental health status from your text")
st.markdown(f"**Detectable conditions:** {', '.join(classes)}")
st.markdown("---")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_area(
    "Type your statement here:",
    height=100,
    placeholder="Example: I feel overwhelmed with work and can't sleep properly..."
)

col1, col2 = st.columns([1, 5])
with col1:
    predict_button = st.button("🔮 Predict", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear History")

if clear_button:
    st.session_state.chat_history = []
    st.rerun()

if predict_button and user_input.strip():
    cleaned_input = advanced_clean_text(user_input)
    
    if cleaned_input.strip():
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_classes = [model.classes_[i] for i in top_3_indices]
        top_3_probs = [probabilities[i] * 100 for i in top_3_indices]
        
        st.session_state.chat_history.append({
            'input': user_input,
            'prediction': prediction,
            'top_3_classes': top_3_classes,
            'top_3_probs': top_3_probs
        })
    else:
        st.warning("⚠️ Your text is too short or contains no meaningful words. Please try again.")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 Prediction History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**You said:** {chat['input'][:150]}{'...' if len(chat['input']) > 150 else ''}")
            
            # Color code by condition
            status_colors = {
                'Normal': '🟢',
                'Depression': '🔴',
                'Suicidal': '🔴',
                'Anxiety': '🟠',
                'Bipolar': '🟠',
                'Stress': '🟡',
                'Personality disorder': '🟠'
            }
            
            emoji = status_colors.get(chat['prediction'], '⚪')
            st.markdown(f"{emoji} **Primary Status:** {chat['prediction']} ({chat['top_3_probs'][0]:.1f}%)")
            
            # Show top 3 predictions
            with st.expander("📊 See all probabilities"):
                for cls, prob in zip(chat['top_3_classes'], chat['top_3_probs']):
                    st.write(f"- {cls}: {prob:.2f}%")
            
            st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses NLP machine learning to detect mental health status in text.")
    st.write("**Detectable Conditions:**")
    for cls in classes:
        st.write(f"- {cls}")
    st.write("\n**Features:**")
    st.write("- Advanced text preprocessing")
    st.write("- Trained on 50K+ samples")
    st.write("- Multiclass classification")
    st.write("- Top-3 predictions with probabilities")
