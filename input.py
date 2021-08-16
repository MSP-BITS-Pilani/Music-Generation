import matplotlib.pyplot as plt
import tensorflow as tf
import pyaudio
import os
import librosa
import random



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
    # split into train, val, test
    pass

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
    # given a waveforrm plot it. if given a list of waveforms, then use subplots
    pass

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
