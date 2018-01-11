# A fairly simple RNN trainer for the Keras_NN_text_generator.py. It is
# character-based, generating text one character at a time rather than
# one word at a time. It is small enough to train on a Mac or PC, but
# just barely (it took 30 hours on mine, after which I decided to run
# it on my university's High Performance Computing Cluster).

# Improved with an additional LSTM layer (two instead of one), 50 epochs
# instead of 20, and batch size of 64 instead of 128. Run on text once, then
# use best weighted .hdf5 file to set weighting in keras_text_generator_2.

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Load text and convert to lowercase.
filename = "kjv_revelation.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# Create mapping of unique characters and integers.
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Get counts of total characters and total "vocab" of characters.
num_chars = len(raw_text)
num_vocab = len(chars)
print("Total characters: ", num_chars)
print("Total vocabulary: ", num_vocab)

# Prepare the dataset of input to output pairs (still encoded as integers).
seq_length = 100
dataX = []
dataY = []
for i in range(0, num_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
num_patterns = len(dataX)
print("Total patterns: ", num_patterns)

# Reshape X to be [samples, time steps, features].
X = numpy.reshape(dataX, (num_patterns, seq_length, 1))

# Normalize (rescale integers to range 0-to-1 for easier
# learning by LSTM which uses sigmoid activation function)
X = X / float(num_vocab)

# One-hot-encode the output variable (convert each of 37
# integers-for-characters into a place on a map of 37 places
# mapped as a list of 36 0s and one 1).
y = np_utils.to_categorical(dataY)

# Define LSTM model as two (instead of one) input layers
# with 256 memory units each and 20 dropout probability.
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))

# Define Dense output layer (softmax activation, cross-entropy
# log loss, and ADAM optimization).
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Use "model checkpointing" to save time by filing improvements
# in loss at the end of each epoch. Later will use the best set
# of weights (lowest loss) to instantiate the generative model.
# Define the checkpoint. Do only once, then delete all the .hdf5
# files except the one with the least loss (renamed "bigger").
filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit model to the data, increased to 50 epochs and decreas batch
# size to 64.
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
