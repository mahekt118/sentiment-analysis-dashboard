import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Title
st.title("Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    if 'sentence' not in data.columns:
        st.error("CSV must contain a 'sentence' column.")
    else:
        st.write("### Uploaded Data")
        st.dataframe(data.head())
        
        # Perform sentiment analysis
        data['Polarity'] = data['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['Predicted Sentiment'] = data['Polarity'].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
        
        # Display results
        st.write("### Sentiment Analysis Results")
        st.dataframe(data[['sentence', 'Predicted Sentiment', 'Polarity']])
        
        # Sentiment distribution
        sentiment_counts = data['Predicted Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#4caf50', '#f44336', '#ffeb3b'])
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#4caf50', '#f44336', '#ffeb3b'])
        ax.set_title("Sentiment Breakdown")
        st.pyplot(fig)
else:
    st.write("Upload a CSV file with a 'sentence' column to begin sentiment analysis.")
