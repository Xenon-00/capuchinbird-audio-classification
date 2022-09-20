import os
import tensorflow as tf
import csv
from itertools import groupby
from collections import OrderedDict
from workspace.src.utils import load_mp3_mono

model = tf.keras.models.load_model('model')
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
    
results = {}
for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data','Forest Recordings', file)
    wav = load_mp3_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(16)
    yhat = model.predict(audio_slices)
    results[file] = yhat

class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.999 else 0 for prediction in logits]

postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

final = OrderedDict(sorted(postprocessed.items()))

with open('capuchin-result.csv','w',newline='')as f:
    writer=csv.writer(f,delimiter=',')
    writer.writerow(['clip_name','call_count'])
    for key, value in final.items():
        writer.writerow([key,value])