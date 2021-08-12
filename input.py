import matplotlib.pyplot as plt
import tensorflow as tf
import pyaudio
import os


def extract_data():
    # go through the files and pick out the right directory to use, shuffling, all that messing around
    pass

def load_audio(directory_list):
    # use decode_wav from the audio submodule to read *.wav files
    # get the list of files and read them into a tensor
    # split into train, val, test
    pass

def play_sample(filename):
    # pyaudio seems to be better than playsound (which we discussed because its not been maintained in a while)
    # given a file name make it play the sample
    pass

def visualize_waveform(wave):
    # given a waveforrm plot it. if given a list of waveforms, then use subplots
    pass

