"""
Canonical audio-stream parameters.

These must agree everywhere: the live pipeline, the simulation runner, and the
virtual-clock timing math (one buffer = BUFFER_SIZE / SAMPLE_RATE seconds of
song time). Import from here instead of redefining.
"""

SAMPLE_RATE = 44100
BUFFER_SIZE = 256
