import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string



# Load the model and TF-IDF vectorizer
file_path_1 = r"C:\ML_PRojects_\New folder\Email-SMS Spam classifier\Models\mnb_model.pkl"
with open(file_path_1, 'rb') as model_file:
    model = pickle.load(model_file)


file_path_2 = r"C:\ML_PRojects_\New folder\Email-SMS Spam classifier\Models\tfidf_vectorizer.pkl"
with open(file_path_2, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')


# Initialize the stemmer
ps = PorterStemmer()

# Define text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit app
st.title("Email-SMS Spam Classifier")
st.write("Enter the message to classify it as spam or ham.")

# Input message
input_message = st.text_area("Message")

if st.button("Classify the Message"):
    if input_message.strip():
        # Preprocess the input message
        transformed_message = transform_text(input_message)
        
        # Transform the preprocessed message using the TF-IDF vectorizer
        input_data = tfidf_vectorizer.transform([transformed_message])
        
        # Predict using the model
        prediction = model.predict(input_data)
        
        # Display the result
        if prediction[0] == 1:
            st.error("This message is Spam.")
        else:
            st.success("This message is Ham.")
    else:
        st.warning("Please enter a message to classify.")



















