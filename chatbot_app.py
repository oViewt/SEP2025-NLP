import streamlit as st

st.title("Test App")
st.write("If you see this, Streamlit is working!")

try:
    import pandas
    st.success("✅ pandas loaded")
except:
    st.error("❌ pandas failed")

try:
    import sklearn
    st.success("✅ scikit-learn loaded")
except:
    st.error("❌ scikit-learn failed")

try:
    import nltk
    st.success("✅ nltk loaded")
except:
    st.error("❌ nltk failed")

try:
    import pickle
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Model failed: {e}")
