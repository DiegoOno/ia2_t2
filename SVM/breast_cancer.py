import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

predictors = pd.read_csv('../data/breast_cancer_input.csv')
classification = pd.read_csv('../data/breast_cancer_output.csv')

#Check the data
#print(predictors.shape)
#print(predictors.head())

print('Start training... ')

pTrain, pTest, cTrain, cTest = train_test_split(predictors, classification, test_size=0.25)

svClassifier = SVC(kernel='linear')

print("Running fit function... ")
svClassifier.fit(pTrain, cTrain.values.reshape(-1,))
print("Done\n")

#To make predicts
predictTest = svClassifier.predict(pTest)
#print(predictTest)

predictTest = (predictTest > 0.5)

testAccuracy = accuracy_score(cTest, predictTest)
testRecall = recall_score(cTest, predictTest)
testPrecision = precision_score(cTest, predictTest)

testConfusionMatrix = confusion_matrix(cTest, predictTest)

print('Accuracy: ' + str(testAccuracy))
print('Recall: ' + str(testRecall))
print('Precision: ' + str(testPrecision))

print()
print("Confusion Matrix: ")
print(testConfusionMatrix)
print()

print("Training finished.")

def get_data():

    problem_params = ['o valor medio do raio das celulas,', 'o valor medio da textura das celulas,', 'o valor medio do perimetro das celulas,', 'o valor medio da area das celulas,', 'o valor medio da suavidade das celulas,', 'o valor medio da consistencia das celulas,', 'o valor medio da concavidade das celulas,', 'o valor medio dos pontos de concavidade das celulas,', 'o valor medio da semetria das celulas,', 'o valor medio da dimensão fractal das celulas,', 'o valor do raio com erro padrao,', 'o valor da textura com erro padrao,', 'o valor do perimetro com erro padrao,', 'o valor da area com erro padrao,', 'o valor da suavidade com erro padrao,',
                      'o valor da consistencia com erro padrao,', 'o valor da concavidade com erro padrao,', 'o valor dos pontos de concavidade com erro padrao,', 'o valor da semetria com erro padrao,', 'o valor da dimensão fractal com erro padrao,', 'o pior valor do raio de uma celula,', 'o pior valor da textura de uma celula,', 'o pior valor do perimetro de uma celula,', 'o pior valor da area de uma celula,', 'o pior valor da suavidade de uma celula,', 'o pior valor da consistencia de uma celula,', 'o pior valor da concavidade de uma celula,', 'o pior valor dos pontos de concavidade de uma celula,', 'o pior valor da semetria de uma celula,', 'o pior valor da dimensão fractal de uma celula,']
    newRegister = []
    aux = 0

    i = 0
    while i < len(problem_params):
        aux = input('Insira ' + problem_params[i] + ' do tumor: ')
        print('\n')
        if isnumber(aux):
            newRegister.append(float(aux))
            i = i + 1
        else:
            print('Voce nao inseriu um valor numerico. Insira um novo valor.\n')
    print('\n')
    np_new_register = np.array([newRegister])
    return np_new_register


def isnumber(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def main():
    #newRegister = get_data()

    newRegister = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500,
                             145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

    print("Input: ")
    print(newRegister)
    print()

    result = svClassifier.predict(newRegister)
    print('Valor calculado pela SVM: %f\n' % result)
    result = (int(result) == 1)

    if result:
        print("Seu tumor é maligno.\n")
    else:
        print("Seu tumor é benigno.\n")


if (__name__ == '__main__'):
    main()

