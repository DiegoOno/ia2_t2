import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

predictors = pd.read_csv('../data/breast_cancer_input.csv')
classification = pd.read_csv('../data/breast_cancer_output.csv')

# Check the data
# print(predictors.shape)
# print(predictors.head())

print('Start training... ')

pTrain, pTest, cTrain, cTest = train_test_split(
    predictors, classification, test_size=0.25)

svClassifier = SVC(kernel='linear')

print("Running fit function... ")
svClassifier.fit(pTrain, cTrain.values.reshape(-1,))
print("Done\n")

# To make predicts
predictTest = svClassifier.predict(pTest)
# print(predictTest)

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

    # Registros para não ter que inserir dado por dado
    # Record 0 -> Answer = True
    # newRegister = np.array([[13.08, 15.71, 85.63, 520, 0.1075, 127, 0.04568, 0.0311, 0.1967, 0.06811, 0.1852, 0.7477, 1383, 14.67,
    #                         0.004097, 0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 0.1312, 0.2776, 189, 0.07283, 0.3184, 0.08183]])
    # Record 1 -> Answer = False
    # newRegister = np.array([[13.48, 20.82, 88.4, 559.2, 0.1016, 0.1255, 0.1063, 0.05439, 172, 0.06419, 213, 0.5914, 1545, 18.52,
    #                         0.005367, 0.02239, 0.03049, 0.01262, 0.01377, 0.003187, 15.53, 26.02, 107.3, 740.4, 161, 0.4225, 503, 0.2258, 0.2807, 0.1071]])
    # Record 2 -> Answer = False
    #newRegister = np.array([[17.57, 15.05, 115, 955.1, 0.09847, 0.1157, 0.09875, 0.07953, 0.1739, 0.06149, 0.6003, 0.8225, 4655, 61.1,
    #                         0.005627, 0.03033, 0.03407, 0.01354, 0.01925, 0.003742, 20.01, 19.52, 134.9, 1227, 0.1255, 0.2812, 0.2489, 0.1456, 0.2756, 0.07919]])
    # Recort 3 -> Answer = True
    # newRegister = np.array([[11.46, 18.16, 73.59, 403.1, 0.08853, 0.07694, 0.03344, 0.01502, 0.1411, 0.06243, 0.3278, 1059, 2475, 22.93,
    #                         0.006652, 0.02652, 0.02221, 0.007807, 0.01894, 0.003411, 12.68, 21.61, 82.69, 489.8, 0.1144, 0.1789, 0.1226, 0.05509, 0.2208, 0.07638]])
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
