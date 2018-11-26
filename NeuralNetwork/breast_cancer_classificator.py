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

    #newRegister = get_data()
    newRegister = np.array([[13.08,15.71,85.63,520,0.1075,127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,189,0.07283,0.3184,0.08183]])

    print(newRegister)

    result = classificator.predict(newRegister)
    print('Valor calculado pela rede: %.24f\n' % result[0][0])
    result = (result > 0.5)

    if result:
        print("Seu tumor é maligno.\n")
    else:
        print("Seu tumor é benigno.\n")


if (__name__ == '__main__'):
    main()
