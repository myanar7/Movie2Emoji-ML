# Movie2Emoji-ML
This is a basic machine learning model to predict related emojis from movie summaries.


```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, SpatialDropout1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
``` 
# Veri kümesi oluşturma

```
film_ozetleri = ["Hızlı ve Öfkeli serisi, arabalarla yapılan yarışları konu alır.", "Kara Şövalye, Batman'in Joker ile mücadelesini anlatır.", "Harry Potter, Hogwarts büyücülük okulundaki maceraları konu alır.", "Baba filminde, aile içindeki güç mücadelesi işlenir.", "The Avengers, süper kahramanların bir bir araya gelerek dünyayı kurtarmak için mücadelesini konu alır."]

emojiler = ['🏎️', '🦇', '🧙‍♂️', '👨‍👩‍👧‍👦', '🦸‍♂️']
```
# Film özetleri ve emojileri birleştirerek veri kümesi oluşturma

```
veri = pd.DataFrame({'ozet': film_ozetleri,
'emoji': emojiler})

```

# Özetleri belirteçlere dönüştürme ve belirteç vektörlerini oluşturma

```
tokenizer = Tokenizer()
tokenizer.fit_on_texts(veri['ozet'])
ozetler = tokenizer.texts_to_sequences(veri['ozet'])
ozetler = pad_sequences(ozetler)
```

# Veri kümesini eğitim ve test kümelerine ayırma

```
X_train, X_test, y_train, y_test = train_test_split(ozetler, veri['emoji'], test_size=0.2, random_state=42)
```
# Yapay sinir ağı modeli oluşturma

```
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emojiler), activation='softmax'))
```
# Modeli derleme ve eğitim
```
adam = Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```
# Modeli değerlendirme

```
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss: {:.2f}, Test accuracy: {:.2f}%".format(loss, accuracy * 100))
```
# Tokenizer'ı oluşturma

```
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
```
# Kelimeleri sayısal verilere dönüştürme

```
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```

# Eğitim ve test verilerini aynı boyuta getirme

```
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
```
# Modeli oluşturma

```
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```
# Modeli derleme

```
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```
# Modeli eğitme

```
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
```
# Modelin performansını görselleştirme

```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
# Modelin test verisi üzerindeki performansını hesaplama

```
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {score[0]}')
print(f'Test Accuracy: {score[1]}')
```

