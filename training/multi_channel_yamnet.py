import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from collections import deque
import time
from scipy import stats
from scipy.spatial import distance


print('loading model..')
model = hub.load('https://tfhub.dev/google/yamnet/1')
print('loaded model successfully')

sample_rate = 44100  # Sample rate of the audio
block_size = 256 * 8 # Block size for audio processing
click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(block_size) / block_size * sample_rate / 3000.)

p = pyaudio.PyAudio()
# Start the audio stream from the microphone
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                input_device_index=3,
                rate=sample_rate,
                input=True,
                frames_per_buffer=block_size)

# Start the stream
stream.start_stream()
stream_out = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    output_device_index=1,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=block_size)

stream_out.start_stream()
start_time = time.time()
print('started')

time_sec = 5
num_blocks_per_sec = round(sample_rate / block_size)
num_blocks_per_100ms = int(num_blocks_per_sec / 10)
assert num_blocks_per_100ms > 1
elements_per_sec = block_size * 2 * num_blocks_per_sec
buffer_size = elements_per_sec * time_sec
audio_buffer = list()
embeddings_buffer = list()
all_similarities = deque(maxlen=100)

best_last_seen = 0
flag_count = 0
best_last_seen_ts = time.time()
flag_count_ts = time.time()


def detect_outliers_mean_std(full_data, test_data, std_threshold=3):
    elements = np.array(full_data)
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    outliers = [x for x in test_data if ((x < mean - std_threshold * sd) or (x > mean + std_threshold + sd))]
    return outliers


def detect_outliers_mad(full_data, test_data, threshold=2.5):
    median = np.median(full_data)
    deviations = np.abs(full_data - median)
    mad = np.median(deviations)
    modified_z_scores = 0.6745 * (test_data - median) / mad
    outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
    return outliers


def detect_outliers(full_data, test_data):
    #return detect_outliers_mean_std(full_data, [test_data], std_threshold=2)
    return detect_outliers_mad(full_data, test_data, threshold=2.5)


def apply_yamnet(in_data):
    global audio_buffer, embeddings_buffer, max_last_seen, all_similarities, best_last_seen_ts, best_last_seen, flag_count, flag_count_ts
    # Convert audio data to numpy array
    audio_data_playback = np.copy(np.frombuffer(in_data, dtype=np.float32))
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    audio_data = audio_data / np.iinfo(np.int16).max
    audio_buffer += audio_data.tolist()
    total_num_elements = elements_per_sec
    previous_data = audio_buffer[-1 * total_num_elements:]

    now = time.time()
    scores, embedding, spectrogram = model(previous_data)
    embedding = tf.reduce_mean(embedding, axis=0)
    embeddings_buffer.append(embedding)

    lookback_index = num_blocks_per_sec * 2
    if len(embeddings_buffer) > lookback_index:
        count = 0
        similarities = []
        while count <= lookback_index:
            previous_embedding = embeddings_buffer[-1 * (count + 1)]
            #similarity = distance.minkowski(embedding.numpy(), previous_embedding.numpy(), 2)
            similarity = abs(float(tf.keras.losses.cosine_similarity(previous_embedding, embedding)))
            similarities.append(abs(float(similarity)))
            count += num_blocks_per_100ms

        #similarity = distance.minkowski(embedding.numpy(), previous_embedding.numpy(), 2)
        #all_similarities += similarities
        best_sim = min(similarities)
        all_similarities.append(best_sim)
        outliers = detect_outliers(all_similarities, best_sim)
        print(f"{now - start_time:<3.3f}, similarity: {best_sim}, best_last_seen={best_last_seen}, outliers={outliers}")
        best_last_seen = min(best_last_seen, best_sim)
        if time.time() - best_last_seen_ts > 3:
            best_last_seen_ts = time.time()
            best_last_seen = 1
        if time.time() - flag_count_ts > 1:
            flag_count_ts = time.time()
            flag_count = 0
        if len(outliers) > 0:
            flag_count += 1
        if flag_count > 5:
            flag_count = 0
            audio_data_playback += click_sound
            print("Meaningful change detected!")
            # Perform further actions or analysis here

    if len(audio_buffer) > total_num_elements * 2:
        del audio_buffer[:total_num_elements * -1]

    if len(embeddings_buffer) > lookback_index * 2:
        del embeddings_buffer[:lookback_index * -1]

    stream_out.write(audio_data_playback.tobytes())


class Stream:
    def __init__(self):
        pass


# Callback function for audio stream
while True:
    in_data = stream.read(block_size, exception_on_overflow=False)
    apply_yamnet(in_data)


stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()
