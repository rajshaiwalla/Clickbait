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

#from numpy.core.umath_tests import inner1d

mwe = MWETokenizer()
lemma = WordNetLemmatizer()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

X = vectorizer.fit_transform(X1)

f_names = vectorizer.get_feature_names()

dense = X.todense().tolist()

new_df = pd.DataFrame(dense,columns=f_names)

final_df = pd.concat([new_df,y1],1,sort=False)
final_df.head()

train = final_df.drop(columns="Clickbait")
test = final_df["Clickbait"]

X1train,X1test,y1train,y1test = tts(train,test,test_size = 0.3,random_state=54)

print(X1train.shape[0])

rf = RandomForestClassifier()
model_rf = rf.fit(X1train,y1train)
y_pred_rf = model_rf.predict(X1test)

print(classification_report(y1test,y_pred_rf))

print(accuracy_score(y1test,y_pred_rf))

pickle.dump(rf, open('rf_model.pkl','wb'))

logreg = LogisticRegression(solver='lbfgs')
model_logreg = logreg.fit(X1train,y1train)
y_pred_logreg = model_logreg.predict(X1test)

print(classification_report(y1test,y_pred_logreg))

print(accuracy_score(y1test,y_pred_logreg))

# Saving model to disk
pickle.dump(logreg, open('logreg_model.pkl','wb'))


dl_model = Sequential()
node = 512
dl_model.add(Dense(node,input_dim=16454,activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(node,input_dim=node,activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(node,input_dim=node,activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(node,input_dim=node,activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(node,input_dim=node,activation='relu'))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(2, activation='softmax'))
dl_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
np.random.seed(5)

dl_model.fit(X1train, y1train, 
          validation_data=(X1test, y1test),
          epochs = 30,
          batch_size = 180)

score = dl_model.evaluate(X1test, y1test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pickle.dump(dl_model, open('dl_model.pkl','wb'))