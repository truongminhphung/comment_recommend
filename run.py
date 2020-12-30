import re
from  underthesea import word_tokenize
import pandas as pd 
import matplotlib.pyplot as plt
import emoji
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dt = pd.read_csv('D:/NLP/data_nlp.csv').values

# negative = []
# positive = [] 
# neutral  = []
# for item in dt:
#     if item[1]== 0.:
#        negative.append(item[1])
#     if item[1]== 2.: 
#        positive.append(item[1])
#     else:
#        neutral.append(item[1])

# len_negative = len(negative)
# len_positive = len(positive)
# len_neutral= len(neutral)


# plt.bar("Negative",len_negative,color="blue",label="negative",width=0.4)
# plt.bar("Positive",len_positive,color="red",label="positive",width=0.4)
# plt.bar("Neutral",len_neutral,color="yellow",label="neutral",width=0.4)
# plt.legend(loc="best")
# plt.title("General data obtained")
# plt.xlabel("Attitude")
# plt.ylabel("Number")
# plt.show()


dt = pd.DataFrame(data=dt,columns=['review','label'])

dt_train,dt_test,label_train,label_test = train_test_split(dt.iloc[:,:-1],dt.iloc[:,1],train_size=0.8)

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


from gensim.models import Word2Vec
input_gensim = []
for review in dt_train['review']:
   input_gensim.append(tachtu(review))
    
model = Word2Vec(input_gensim, size=128, window=5, min_count=0, workers=4, sg=1)
model.wv.save("D:/NLP/word.model")

import gensim.models.keyedvectors as word2vec
model_embedding = word2vec.KeyedVectors.load('D:/NLP/word.model')

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

train_data = []
label_data = []

for x in tqdm(dt_train['review']):
   train_data.append(comment_embedding(x))
train_data = np.array(train_data)

for y in tqdm(label_train):
   label_ = np.ones(3)
   if int(y) == 0:
      label_=[1,0,0]
   elif int(y) == 1:
      label_ = [0,1,0]
   else:
      label_ = [0,0,1]
   label_data.append(label_)

import numpy as np
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
from keras.preprocessing import sequence

sequence_length = 200
embedding_size = 128
num_classes = 3
filter_sizes = 3
num_filters = 150
epochs = 50
batch_size = 30
learning_rate = 0.01
dropout_rate = 0.5

x_train = train_data.reshape(train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
y_train = np.array(label_data)

# Define model
model = keras.Sequential()
model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
                        padding='valid',
                        input_shape=(sequence_length, embedding_size, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(198, 1)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
# Train model
adam = tf.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(model.summary())

model.fit(x = x_train[:4500], y = y_train[:4500], batch_size = batch_size, 
verbose=1, epochs=epochs, validation_data=(x_train[:1865], y_train[:1865]))

model.save('models.h5')