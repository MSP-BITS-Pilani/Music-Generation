# Change Log

###18-Aug-2021 : updated visualize_data in input.py, Arvind
Modified code in visualize_data to plot a graph for a given .wav file using librosa. TODO: Decide what to pass into conv layers.

### 16-Aug-2021 : updated input.py, Abhigyan###
Added code to the functions load_audio to input a filepath containing .wav files and return a list of tensors containing decoded float32 np arrays. Added functionality to play_sample using PyAudio that takes in a filename and plays the corresponding .wav audio. Made functional temporarily visualize_waveform using tensorflow(to be updated) and modified parameters of extract_data to include an option of instrument of choice. ToDo Add a readme file listing allowed inputs for the same.   

### 13-Aug-2021 : extract_data and (new)load_data_mono(incomplete), Abhay
Added code to 'extract_data' to store files of a certain instrument into a list, and shuffle them. Added a new function load_data_mono in order to work with mono music instead of stereo - a decision we will have to make. The code is still incomplete, have to decide on what exactly needs to be done - in what format Wavenet takes input. 
### 12-Aug-2021 : initial files, Balan
Added the `input.py` and `train.py` files with a rudimentary structure of functions to be implemented. More functions can be added if necessary. Defined classes for two models - VAE and WaveNet, to be implemented later. Download the IRMAS dataset from its mirror at [zenodo](https://zenodo.org/record/1290750#.YRQW1XUzbmx) (only the training set).

### {date} : {commit name}, {your name}
  {information about the commit, what\'s new how it affects other code (if it does)}
