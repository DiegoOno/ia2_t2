import numpy as np
from keras.models import model_from_json

# Load a json file with neural network configuration generated in training step
classificator_file = open('classificator_breast.json', 'r')
classificator_sctructure = classificator_file.read()
classificator_file.close()
classificator = model_from_json(classificator_sctructure)

classificator.load_weights('classificator_breast_weights.h5')

newRegister = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

result = classificator.predict(newRegister)

if result > 0.7:
    print("Seu tumor é maligno.\n")
else:
    print("Seu tumor é benigno.\n")