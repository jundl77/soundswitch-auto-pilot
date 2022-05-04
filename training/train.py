#! /usr/bin/env python

import aubio
import numpy as np
from sklearn.cluster import KMeans
import datetime


def run(path):
    sample_rate, win_s, hop_s = 44100, 1024, 512
    mfcc_filters = 40  # must be 40 for mfcc
    mfcc_coeffs = 13

    aubio_source = aubio.source(path, sample_rate, hop_s)
    onset_o: aubio.onset = aubio.onset("default", win_s, hop_s, sample_rate)
    tempo_o: aubio.tempo = aubio.tempo("default", win_s, hop_s, sample_rate)
    pvoc_o: aubio.pvoc = aubio.pvoc(win_s, hop_s)
    mfcc_o: aubio.mfcc = aubio.mfcc(win_s, mfcc_filters, mfcc_coeffs, sample_rate)
    energy_filter = aubio.filterbank(40, win_s)
    energy_filter.set_mel_coeffs_slaney(sample_rate)

    mfccs = np.zeros([mfcc_coeffs, ])
    energies = np.zeros((40,))

    total_frames = 0
    while True:
        samples, read = aubio_source()

        #is_onset: bool = onset_o(samples)[0] > 0
        #if is_onset:
        if total_frames % (hop_s * 10) == 0:
            spec = pvoc_o(samples)
            new_energies = energy_filter(spec)
            mfcc_out = mfcc_o(spec)
            mfccs = np.vstack((mfccs, mfcc_out))
            energies = np.vstack([energies, new_energies])

        is_beat = tempo_o(samples)

        total_frames += read
        if read < hop_s:
            break

    print('starting kmeans')
    kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(mfccs)
    return tempo_o.get_bpm()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sources',
            nargs='+',
            help="input_files")
    args = parser.parse_args()
    for f in args.sources:
        bpm = run(f)
        print("{:6s} {:s}".format("{:2f}".format(bpm), f))