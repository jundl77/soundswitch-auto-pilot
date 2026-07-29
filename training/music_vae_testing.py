import pyaudio
import numpy as np
from magenta.models.music_vae import TrainedModel
import magenta.music as mm

model = TrainedModel(model_name='hierdec-mel_16bar', batch_size=4, checkpoint_dir_or_path='PATH_TO_CHECKPOINT')

sample_rate = 44100
block_size = 2048
threshold = 0.5

previous_embedding = None

click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(block_size) / block_size * sample_rate / 3000.)

p = pyaudio.PyAudio()
stream_out = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    output_device_index=12,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=block_size)


def audio_callback(in_data, frame_count, time_info, status):
    global previous_embedding

    audio_data = np.frombuffer(in_data, dtype=np.float32)

    spectrogram = mm.audio_io.wav_data_to_mel_spectrogram(audio_data, sample_rate)

    reshaped_input = np.expand_dims(spectrogram, axis=0)

    embeddings = model.encode(reshaped_input)[0]

    if previous_embedding is not None:
        similarity = np.linalg.norm(previous_embedding - embeddings)
        if similarity > threshold:
            in_data += click_sound
            print("Meaningful change detected!")

    previous_embedding = embeddings
    stream_out.write(in_data)

    return None, pyaudio.paContinue

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=block_size,
                stream_callback=audio_callback)

stream.start_stream()
stream_out.start_stream()

try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()

p.terminate()
