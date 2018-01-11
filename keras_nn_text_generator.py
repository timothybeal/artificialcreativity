# Generates text for KJV Revelation with improved layer topology and
# and more epochs to get better weight file. Run only after checkpointing 
# and fitting the model using the first program (keras_nn_text_gen_trainer.py). Here, 
# load weights by replacing those checkpoint and fit lines with the best 
# (least loss) -bigger.hdf5 file.

# Add import sys and remove import keras.callbacks.ModelCheckpoint.
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
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
print(chars)

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

# Define LSTM model as single input layer with 256 memory units
# and .20 dropout probability. Define Dense output layer 
# (softmax activation).
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# Load network weights (after checkpoint and fit on first run).
# Compile model with cross-entropy and ADAM optimization.
filename = "weights-improvement-41-0.6912-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Choose a random seed.
start = numpy.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate characters.
print("Generated:")
for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(num_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
