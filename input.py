import matplotlib.pyplot as plt
import tensorflow as tf
import pyaudio
import os
import librosa, librosa.display
import random
import numpy as np



def extract_data(instr):
    # go through the files and pick out the right directory to use, shuffling, all that messing around
    path = 'G:\My Drive\Music Generation\IRMAS-TrainingData\\'+instr
    #Using a placeholder path that consists of (mainly) a single instrument to avoid the complexity of polyphonic music. Will have to be decided upon later.
    ### Add a function parameter asking which instrument to use, you can also provide a list of allowed strings in the readme or sth
    file_names = []
    for file in os.listdir(path):
        file_names.append(path + '\\' + file)
    random.shuffle(file_names)
    #Technically redundant, since the file names are appended in an arbitrary order
    return file_names
    #TODO:    Decide whether we want to use a single instrument. 

def load_audio(directory_list):
    # use decode_wav from the audio submodule to read *.wav files
    # get the list of files and read them into a tensor
    # split into train, val, test (is there a need since the dataset already has multiple testing sets)
    waveforms=list()
    for path in directory_list:
        parts=tf.strings.split(path, os.path.sep)
        label=parts[-2]                  #Are labels needed? Since we will be shuffling data randomly.
        audio_binary = tf.io.read_file(path)
        waveform, sr = tf.audio.decode_wav(audio_binary)
        waveforms.append(waveform)
        
    return waveforms
    

def play_sample(filename):
    # pyaudio seems to be better than playsound (which we discussed because its not been maintained in a while)
    # given a file name make it play the sample
    chunk_size = 1024
    wf = wave.open(filename,'rb')
    interface = pyaudio.PyAudio()
    stream = interface.open(format=interface.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
    
    aud_data = wf.readframes(chunk_size)
    
    while len(aud_data)>0:
        stream.write(aud_data)
        aud_data = wf.readframes(chunk_size)
    
    stream.stop_stream()
    stream.close()
    wf.close()
    interface.terminate()
    

def visualize_waveform(wave):
   #plot a graph for a given .wav file
    x,sr=librosa.load(wave)
    librosa.display.waveplot(x, sr)
    
def load_audio_mono(file_list):
    # In case we are using mono music (1 channel) instead of stereo  (2 channels), since it is easier, and the vector size is halved.
    ### use mono for now, if needed, we can implement stereo as well
    tensor_list = []
    for song in file_list:
        x, sr = librosa.load(song, sr=22050) 
        tensor_list.append[x]
    # TODO: Split each tensor into smaller samples? The audio clips themselves are only 3 seconds long, so there is barely any temporal dependency.
    #       Split into train,val,test - if using a certain instrument(?) it can be done manually, otherwise train_test_split twice.
    #       Make use of tf.data.Datasets for convenience in batching and the like.
    #       Figure out what exactly we are passing to the conv layers and in what shape.       
    pass


def audio_to_mel(list):
    mel_spec_list = []
    for sample in list:
        spect = librosa.feature.melspectogram(y = sample)
        mel_spec_list.append(spect)
    return mel_spec_list

def get_mean_and_std(spec_list):
    means = []
    stds = []
    for spec in spec_list:
        means.append(np.mean(spec, axis = 1))
        stds.append(np.std(spec, axis = 1))
    mean = np.mean(means, axis = 0)
    std = np.mean(stds, axis = 0)
    return mean, std

def normalize_mel(spectogram_list, spec_mean, spec_std):
    norm_spec = []
    for item in spectogram_list:
        norm_item = (item - spec_mean) / (3.0 * spec_std)  
        clipped = np.clip(norm_item, -1.0, 1.0)
        norm_spec.append(clipped)
    return norm_spec


def denormalize_mel(melspectogram_list, mean_spec, std_spec):
    denorm_spec = []
    for gram in melspectogram_list:
        denorm_gram = (gram * (3.0 * std_spec)) + mean_spec
        denorm_spec.append(denorm_gram)
    return denorm_spec

def load_spec(mel_list):
    append = np.ones((128, 128)) * (-80)
    right_size_mel = []
    for melspec in mel_list:
        melspec = np.hstack((melspec,append))
        melspec = melspec[:, :128]
        right_size_mel.append(melspec)
    return right_size_mel
        


def mel_to_audio(spec):
    pass


def plot_spectograms(melspec_list):
    #If we want to see the denormalized spectograms for some reason. Here 9 will be shown. Changeable.
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(melspec_list[i])
    plt.show()
