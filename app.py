import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string
import nltk
nltk.download('punkt_tab')

ps = PorterStemmer()

import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv',encoding='latin1')
df.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# make the column names descriptive
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')

#create three new columns
df['num_char'] = df['text'].apply(len)

# number of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

#number of sentences
df['num_sent'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    #only alphabetic or numeric characters
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # removing stopwords , punctuations
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

df['transformed_text'] = df['text'].apply(transform_text)

from sklearn.feature_extraction.text  import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf2 = TfidfVectorizer()

X = tfidf2.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=2,test_size=0.2)

from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

mnb = MultinomialNB()

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))

mnb.fit(x_train,y_train)

st.title('Email/Spam Classifier')

sms = st.text_input('Enter the message')

#pre - trained or fit with the spam.csv data

# pre-processing
transformed_sms = transform_text(sms)
# 2. vectorize
vector_input = tfidf2.transform([transformed_sms])
# 3. predict
result = mnb.predict(vector_input)[0]

if st.button('predict'):
    # resutls
    if result == 1:
        st.header('Spam')
    else:
        st.header('No Spam')
