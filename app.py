import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('names')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
english_stopwords = stopwords.words('english')

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.layers import  Dropout, Dense
from tensorflow.python.keras.models import Sequential

from flask import Flask,render_template,url_for,request

#from numpy.core.umath_tests import inner1d

mwe = MWETokenizer()
lemma = WordNetLemmatizer()


app = Flask(__name__)
logreg = pickle.load(open('logreg_model.pkl', 'rb'))


def remove_stopwords(text):
  token = text.split()
  return ' '.join([w for w in token if not w in english_stopwords])

def remove_punctuation(text):
  text = [i for i in text if i.isalpha() or i.isspace()]
  return ''.join(text)

def shortwords(text):
  text = ' '.join([w for w in text.split() if len(w)>2])
  return text

def tokenize(text):
  return mwe.tokenize(text)

def lemmatize(text):
  return ''.join([lemma.lemmatize(word,'v') for word in mwe.tokenize(text)])


def preprocess(s):
  s = s.lower()
  text = remove_stopwords(s)
  text = shortwords(text)
  text = remove_punctuation(text)
  text = tokenize(text)
  text = lemmatize(text)
  
  return text

vectorizer = TfidfVectorizer(max_features=75000)

df = pd.read_csv("data.csv")
df.head()

df.Text = df.Text.apply(preprocess)

df.head()

X1 = df.Text
y1 = df.Clickbait

vectorizer.fit(X1)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	if request.method == 'POST':
		Headline = request.form['Headline']
		Headline = preprocess(Headline)
		data = [Headline]
		X = vectorizer.transform(data).toarray()
		my_prediction = logreg.predict(X)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

