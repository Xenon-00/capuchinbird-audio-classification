import tensorflow as tf
import tensorflow_io as tfio

def load_wav_mono(filepath):
    file_contents = tf.io.read_file(filepath)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out = 16000)
    return wav

def load_mp3_mono(filepath):
    res = tfio.audio.AudioIOTensor(filepath)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1)/2
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def convert_to_spectrogram(filepath, label):
    wav = load_wav_mono(filepath)
    wav = wav[:48000]
    padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label