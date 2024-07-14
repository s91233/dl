#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, datetime, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense


# ## Tokenize

# In[ ]:


with open("./pg1513.txt", 'r', encoding='utf-8') as file: text = file.read()
DIR = f"{os.path.splitext(os.path.basename(file.name))[0]}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs('./models/'+DIR, exist_ok=True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
dict_size = len(tokenizer.word_index) + 1

sequences = np.array( pad_sequences( [tokenizer.texts_to_sequences([line])[0][:i] for line in text.split('\n') for i, _ in enumerate(tokenizer.texts_to_sequences([line])[0], start=1)], padding='pre') )

Y = np.array(tf.keras.utils.to_categorical(sequences[:, -1], num_classes=dict_size))

with open("./models/{0}/dict.json".format(DIR), 'w', encoding='utf-8') as f:
    json.dump({
        "word_index": tokenizer.word_index,
        "index_word": tokenizer.index_word,
        "sequence_l": sequences.shape[1]
    }, f, ensure_ascii=False
)


# ## Train

# In[ ]:


model = Sequential()
model.add(Embedding(dict_size, 100, input_shape=(sequences.shape[1]-1,)))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(dict_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
model.summary()

history = model.fit(sequences[:, :-1], Y, epochs=1, batch_size=32, verbose=2, callbacks=[TensorBoard()])

model.save('./models/{0}/model.keras'.format(DIR))


# ## Test

# In[11]:


test = "Henceforth"

for _ in range(10): test += " " + tokenizer.index_word.get( np.argmax( model.predict( pad_sequences([(tokenizer.texts_to_sequences([test])[0])], maxlen=sequences.shape[1] - 1, padding='pre'), verbose=0) ), "")

print(test)


# ## Export

# In[12]:


import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "./models/{0}/tfjs".format(DIR))