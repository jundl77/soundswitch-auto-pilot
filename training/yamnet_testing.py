import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import csv
import time
from pathlib import Path
from scipy.spatial import distance


def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

# Load YAMNet pre-trained model
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# Parameters
sample_rate = 44100  # Sample rate of the audio
block_size = 1024 * 64   # Block size for audio processing

# Variables
previous_embedding = None

click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(block_size) / block_size * sample_rate / 3000.)

p = pyaudio.PyAudio()
# Start the audio stream from the microphone
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                input_device_index=4,
                rate=sample_rate,
                input=True,
                frames_per_buffer=block_size)

# Start the stream
stream.start_stream()
stream_out = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    output_device_index=12,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=block_size)

stream_out.start_stream()
start_time = time.time()
print('started')

# Callback function for audio stream
while True:
    in_data = stream.read(block_size, exception_on_overflow=False)

    # Convert audio data to numpy array
    audio_data_playback = np.copy(np.frombuffer(in_data, dtype=np.float32))
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    audio_data = audio_data / np.iinfo(np.int16).max

    scores, embedding, spectrogram = model(audio_data)
    embedding = tf.reduce_mean(embedding, axis=0)

    scores = scores.numpy()
    spectrogram = spectrogram.numpy()
    infered_class = class_names[scores.mean(axis=0).argmax()]

    top_classes = np.argsort(scores.mean(axis=0))[::-1]
    top = "|".join(f"{class_names[i]:<15}" for i in top_classes[:5])

    # Check for change in embeddings
    if previous_embedding is not None:
        similarity = abs(tf.keras.losses.cosine_similarity(previous_embedding, embedding))
        #similarity = distance.minkowski(embedding.numpy(), previous_embedding.numpy(), 2)
        print(f"{time.time() - start_time:<3.3f},{infered_class}, similarity: {similarity}")
        if similarity > 6:
            audio_data_playback += click_sound
            print("Meaningful change detected!")
            # Perform further actions or analysis here

    # Update previous_embedding for the next block
    previous_embedding = embedding
    stream_out.write(audio_data_playback.tobytes())


stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()
