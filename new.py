import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from mltk.stem import wordNetLemmatizer

Lemmatizer = wordNetLemmatizer()

intents = json.loads(open('intens.json').read())

words = []
classes = []
documents = []
ignoreletters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [Lemmatizer.lemmatize(word) for word in words if word not in ignoreletters]
words = sorted(set(classes))

classes = sorted(set(classes))

pickle.dump(words, open('words.pki','wb'))
pickle.dump(classes,open('classes.pki','wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [Lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag+outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:,:len(words)]
trainY = training[:,len(words):]

model = tf.keras.Sequential()

model.add(tf.keras.layer.Dense(128,input_shape = (len(trainX[0]),),activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]),activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01,momentum = 0.9,nesterov = True)

model.compile(loss = 'categorical_crossentropy',optimaizer = sgd,metrics = ['accuracy'])
hist = model.fit(np.array(trainX),np.array(trainY),epochs = 200 , batch_size = 5,verbose = 1)
model.save('chatbot_simplilearnmodel.h5',hist)
print('Executed')