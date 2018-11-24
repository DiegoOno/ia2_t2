import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Read input and output files
predictors = pd.read_csv('../data/breast_cancer_input.csv')
classes = pd.read_csv('../data/breast_cancer_output.csv')

predictors_training, predictors_test, training_classes, training_test = train_test_split(predictors, classes, test_size = 0.25)

# Create a neural network
classificator = Sequential()

# Add one hidden layer with Dropout
classificator.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificator.add(Dropout(0.2))

# Add another hidden layer with Dropout
classificator.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'normal'))
classificator.add(Dropout(0.2))

# Add output layer
classificator.add(Dense(units = 1, activation = 'sigmoid'))

# Configure the learning process
classificator.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# Training
classificator.fit(predictors_training, training_classes, batch_size = 10, epochs = 100)

# Save network structure in a json file
classificator_json = classificator.to_json()
with open('classificator_breast.json', 'w') as json_file:
    json_file.write(classificator_json)

# Save the weights of network after training
classificator.save_weights('classificator_breast_weights.h5')