from src.utils import load_mp3_mono, load_wav_mono, convert_to_spectrogram
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import os
import joblib

# Avoid OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

positive_path = os.path.abspath(os.path.join("data", "Parsed_Capuchinbird_Clips"))
negative_path = os.path.abspath(os.path.join("data", "Parsed_Not_Capuchinbird_Clips"))

pos = tf.data.Dataset.list_files(positive_path+'\*.wav')
neg = tf.data.Dataset.list_files(negative_path+'\*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

data = data.map(convert_to_spectrogram)
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(round(len(data) * .7))
val = data.skip(round(len(data) * .7)).take(round(len(data) * .3))

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D((3,3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.fit(train, epochs=10, validation_data=val)
model.save('model')
