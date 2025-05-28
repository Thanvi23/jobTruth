import tensorflow as tf
import joblib
from keras.layers import TFSMLayer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
class prediction_model:
    def __init__(self):
        self.tokenizer = joblib.load('./tokinezer_file')
        self.model = TFSMLayer('./bidirectional_model/', call_endpoint='serving_default')

    def preprocess_text(self, text):
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in set(stopwords.words("english"))]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    def predict(self, text):
        # Example input
        input_text = text
        cleaned_text = self.preprocess_text(input_text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)  # use your training maxlen
        prediction = self.model(tf.cast(padded_sequence, tf.float32))
        pred_value = prediction['dense'].numpy()[0][0]
        return pred_value

# predictor=prediction_model()
# text = "This is a sample text for prediction."
# print(predictor.predict(text))
