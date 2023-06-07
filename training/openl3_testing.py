import pyaudio
import numpy as np
import openl3
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenL3 pre-trained model
#model = openl3.load_model()

# Parameters
sample_rate = 44100  # Sample rate of the audio
block_size = 256   # Block size for audio processing
threshold = 0.5     # Threshold for change detection

# Variables
previous_embedding = None

click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(block_size) / block_size * sample_rate / 3000.)

p = pyaudio.PyAudio()
stream_out = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    output_device_index=12,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=block_size)

model = openl3.models.load_audio_embedding_model(input_repr="linear", content_type="env",
                                                 embedding_size=512)

# Callback function for audio stream
def audio_callback(in_data, frame_count, time_info, status):
    global previous_embedding

    # Convert audio data to numpy array
    audio_data = np.frombuffer(in_data, dtype=np.float32)

    # Extract embeddings from the audio block
    embeddings, _ = openl3.get_audio_embedding(audio_data, sample_rate, model=model,
                               input_repr="linear", embedding_size=512)

    # Check for change in embeddings
    if previous_embedding is not None:
        #similarity = True
        similarity = cosine_similarity(previous_embedding.reshape(1, -1), embeddings.reshape(1, -1))
        if similarity < threshold:
            in_data += click_sound
            print("Meaningful change detected!")
            # Perform further actions or analysis here

    # Update previous_embedding for the next block
    previous_embedding = embeddings

    stream_out.write(in_data)

    return None, pyaudio.paContinue


stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                input_device_index=4,
                rate=sample_rate,
                input=True,
                frames_per_buffer=block_size,
                stream_callback=audio_callback)

stream.start_stream()
stream_out.start_stream()

# Keep the stream running until interrupted
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    # Stop the stream
    stream.stop_stream()
    stream.close()

# Terminate PyAudio
p.terminate()
