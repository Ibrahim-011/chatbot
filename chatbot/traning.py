import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract data from JSON
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to files
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Pad bags with zeros to ensure consistent length
max_bag_length = max(len(row[0]) for row in training)
for i in range(len(training)):
    bag, output_row = training[i]
    # Pad bag with zeros if needed
    if len(bag) < max_bag_length:
        bag += [0] * (max_bag_length - len(bag))
    training[i] = [bag, output_row]

# Convert training data to NumPy arrays
training_x = np.array([i[0] for i in training])
training_y = np.array([i[1] for i in training])

# Build neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Define a data generator
def data_generator(training_x, training_y, batch_size):
    num_batches = len(training_x) // batch_size
    while True:
        for i in range(num_batches):
            batch_x = training_x[i * batch_size: (i + 1) * batch_size]
            batch_y = training_y[i * batch_size: (i + 1) * batch_size]
            yield batch_x, batch_y

# Train model using generator
batch_size = 32
steps_per_epoch = len(training_x) // batch_size
hist = model.fit(data_generator(training_x, training_y, batch_size), epochs=200, steps_per_epoch=steps_per_epoch, verbose=1)

# Save model
model.save('model.h5')

print("Model created and saved")
