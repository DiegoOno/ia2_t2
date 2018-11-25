import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Create a neural network
classificator = Sequential()


def save_network():
    # Save network structure in a json file
    classificator_json = classificator.to_json()
    with open('classificator_breast.json', 'w') as json_file:
        json_file.write(classificator_json)

    # Save the weights of network after training
    classificator.save_weights('classificator_breast_weights.h5')


def show_weights():
    weight0 = classificator.layers[0].get_weights()
    weight1 = classificator.layers[1].get_weights()
    weight2 = classificator.layers[2].get_weights()
    print(weight0)
    print(weight1)
    print(weight2)


def main():
    # Read input and output files
    predictors = pd.read_csv('../data/breast_cancer_input.csv')
    classes = pd.read_csv('../data/breast_cancer_output.csv')

    predictors_training, predictors_test, training_class, test_class = train_test_split(
        predictors, classes, test_size=0.25)

    # Add one hidden layer with Dropout
    # input_dim (Used only in first hidden layer) -> Number of input
    # In the first attempt the units of hidden layer are considered as number_of_input + number_of_output
    classificator.add(Dense(units=16, activation='relu',
                            kernel_initializer='random_uniform', input_dim=30))
    classificator.add(Dropout(0.2))

    # Add another hidden layer with Dropout
    classificator.add(Dense(units=8, activation='relu',
                            kernel_initializer='random_uniform'))
    classificator.add(Dropout(0.1))

    classificator.add(Dense(units=8, activation='relu',
                            kernel_initializer='random_uniform'))
    classificator.add(Dropout(0.1))
    # Add output layer
    classificator.add(Dense(units=1, activation='sigmoid'))

    training_optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    # Configure the learning process
    classificator.compile(
        optimizer=training_optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    # Training
    # Batch_size -> Calculates error for n records after this the weights will update
    # Number of times the weights will be ajusted
    classificator.fit(predictors_training, training_class,
                      batch_size=30, epochs=500)

    # Use test data
    predict_test = classificator.predict(predictors_test)

    # Convert the output value for each record to boolean
    predict_test = (predict_test > 0.5)

    # Calculate the accuracy based on test outputs values
    test_accuracy = accuracy_score(test_class, predict_test)

    # Generate confusion matrix
    test_confusion_matrix = confusion_matrix(test_class, predict_test)

    print('Accuracy: ' + str(test_accuracy) + '\n')
    print(test_confusion_matrix)

    if test_accuracy > 0.95:
        print("Network and weight will be saved.\n")
        save_network()
    else:
        print("Insufficient accuracy. Try again!\n")

if (__name__ == '__main__'):
    main()
