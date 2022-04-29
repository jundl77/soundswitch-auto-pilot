#! /usr/bin/env python3.9

from clients.pyaudio_client import PyAudioClient
from clients.midi_client import MidiClient
from analyser.music_analyser import MusicAnalyser
from analyser.music_analyser_handler import MusicAnalyserHandler

buffer_size = 512
sample_rate = 44100


def main():
    audio_client: PyAudioClient = PyAudioClient(sample_rate, buffer_size, output_device_index=1)
    midi_client: MidiClient = MidiClient()
    handler: MusicAnalyserHandler = MusicAnalyserHandler(midi_client)
    music_analyser: MusicAnalyser = MusicAnalyser(sample_rate, buffer_size, handler)

    audio_client.start_streams(start_stream_out=True)

    while True:
        audio_signal = audio_client.read()
        new_audio_signal = music_analyser.analyse(audio_signal)

        if audio_client.support_output():
            audio_client.play(new_audio_signal)

    audio_client.close()


if __name__ == "__main__":
    main()
