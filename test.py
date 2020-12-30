import gensim.models.keyedvectors as word2vec
import numpy as np
import emoji
import re
from  underthesea import word_tokenize
import pandas as pd 
from sklearn.model_selection import train_test_split

dt = pd.read_csv('D:/NLP/data_nlp.csv').values

dt = pd.DataFrame(data=dt,columns=['review','label'])

dt_train,dt_test,label_train,label_test = train_test_split(dt.iloc[:,:-1],dt.iloc[:,1],train_size=0.8,random_state=0)

model_embedding = word2vec.KeyedVectors.load('word.model')

word_labels = []
max_seq = 200
embedding_size = 128

for word in model_embedding.vocab.keys():
    word_labels.append(word)


def comment_embedding(comment):
   matrix = np.zeros((max_seq, embedding_size))
   words = tachtu(comment)
   lencmt = len(words)

   for i in range(max_seq):
      indexword = i % lencmt
      if (max_seq - i < lencmt):
         break
      if(words[indexword] in word_labels):
         matrix[i] = model_embedding[words[indexword]]
   matrix = np.array(matrix)
   return matrix


def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def clean_text(sentence):
   return re.sub('[\^.,!?:\-\(\{\}\)\/]','',sentence).lower()

def tachtu(sentence):
   sentence = clean_text(sentence)
   sentence = give_emoji_free_text(sentence)
   sentence = word_tokenize(sentence)
   return sentence


from keras.models import load_model
model_sentiment = load_model("D:/NLP/models.h5")

predict= []
for i in range(len(dt_test)):
    maxtrix_embedding = np.expand_dims(comment_embedding(dt_test.iloc[i][0]), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    attitue = (model_sentiment.predict(maxtrix_embedding))
    attitue = np.argmax(attitue)
    predict.append(attitue)

result = []
for i in range(len(label_test)):
    result.append(label_test.iloc[i])

from sklearn.metrics import accuracy_score
print(accuracy_score(result, predict))