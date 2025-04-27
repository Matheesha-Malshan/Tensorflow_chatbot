import tensorflow as tf
import preprocess
from sklearn.preprocessing import LabelEncoder
import numpy as np

import matplotlib.pyplot as plt

label_encoding=LabelEncoder()
en_data=label_encoding.fit_transform(preprocess.y)

out_shape=en_data.shape[0]
train_data=preprocess.x
train_labels=en_data[:,np.newaxis]

model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(2,activation="relu"))
model.add(tf.keras.layers.Dense(5,activation="relu"))
model.add(tf.keras.layers.Dense(8,activation="relu"))

model.add(tf.keras.layers.Dense(out_shape,activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(train_data,train_labels,epochs=7,batch_size=4)
model.summary()

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

model.save("chat.h5")


