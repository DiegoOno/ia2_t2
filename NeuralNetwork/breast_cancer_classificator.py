#!/usr/bin/env python
import numpy as np
from keras.models import model_from_json


def get_data():

    problem_params = ['o valor medio do raio das celulas,', 'o valor medio da textura das celulas,', 'o valor medio do perimetro das celulas,', 'o valor medio da area das celulas,', 'o valor medio da suavidade das celulas,', 'o valor medio da consistencia das celulas,', 'o valor medio da concavidade das celulas,', 'o valor medio dos pontos de concavidade das celulas,', 'o valor medio da semetria das celulas,', 'o valor medio da dimensão fractal das celulas,', 'o valor do raio com erro padrao,', 'o valor da textura com erro padrao,', 'o valor do perimetro com erro padrao,', 'o valor da area com erro padrao,', 'o valor da suavidade com erro padrao,',
                      'o valor da consistencia com erro padrao,', 'o valor da concavidade com erro padrao,', 'o valor dos pontos de concavidade com erro padrao,', 'o valor da semetria com erro padrao,', 'o valor da dimensão fractal com erro padrao,', 'o pior valor do raio de uma celula,', 'o pior valor da textura de uma celula,', 'o pior valor do perimetro de uma celula,', 'o pior valor da area de uma celula,', 'o pior valor da suavidade de uma celula,', 'o pior valor da consistencia de uma celula,', 'o pior valor da concavidade de uma celula,', 'o pior valor dos pontos de concavidade de uma celula,', 'o pior valor da semetria de uma celula,', 'o pior valor da dimensão fractal de uma celula,']
    newRegister = []
    aux = 0

    i = 0
    while i < 30:
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
    # Load a json file with neural network configuration was generated in the training step
    classificator_file = open('classificator_breast.json', 'r')
    classificator_sctructure = classificator_file.read()
    classificator_file.close()
    classificator = model_from_json(classificator_sctructure)

    # Load the weight of network that was found in the training
    classificator.load_weights('classificator_breast_weights.h5')

    newRegister = get_data()
    #newRegister = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500,
    #                         145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
    print(newRegister)

    result = classificator.predict(newRegister)
    print('Valor calculado pela rede: %f\n' % result[0][0])
    result = (result > 0.5)

    if result:
        print("Seu tumor é maligno.\n")
    else:
        print("Seu tumor é benigno.\n")


if (__name__ == '__main__'):
    main()
