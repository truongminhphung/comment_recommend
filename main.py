from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import gensim.models.keyedvectors as word2vec
import emoji
import re
from  underthesea import word_tokenize

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


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

model = load_model('models.h5')
@app.route('/predict', methods=['POST'])
def home():
    data = request.form['comment']
    maxtrix_embedding = np.expand_dims(comment_embedding(data), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    pred = model.predict(maxtrix_embedding)
    pred = np.argmax(pred)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)