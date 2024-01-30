# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:57:15 2024

@author: tiago Ventura

Redes Neurais Convolucionais - POO
"""


# Construindo a rede neural convolucional
import tensorflow as tf
import keras as k
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class RedesNeuraisConvolucionais:
    
    def __init__(self, steps_per_epoch, qtd_epoch, steps_validation
                 , folder_dataset_validation, folder_dataset_training):
        
        #Parâmetros utilizado para treinamento.
        self._steps_per_epoch = steps_per_epoch
        self._qtd_epoch = qtd_epoch
        self._steps_validation = steps_validation
        self._folder_dataset_validation = folder_dataset_validation
        self._folder_dataset_training = folder_dataset_training
        
        #Inicializando a rede neural convolucional
        self.classifier = Sequential()
        
    def pooling_layer(self, param1, param2):
        self.classifier.add(MaxPooling2D(pool_size=(param1,param2)))
        
    def add_layer(self, param1, param2, param3):
        self.classifier.add(Conv2D(param1, (param2, param3), activation = 'relu'))                                
        
    def flatter_function(self):
        self.classifier.add(Flatten())
        
        
    def build_rnc(self):
        # Primeira camada de convolução
        self.classifier.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))

        '''
        Aplicando o agrupamento(pooling) para reduzir o tamanho do mapa de features 
        resultada da primeira camada de convolução(dividido por 2)
        '''
        # Passo 2 - Pooling
        self.pooling_layer(2, 2)

        #Segunda camada de convolução
        self.add_layer(32, 3, 3)

        #Aplicação a camada de pooling à saída da camada de convolução anterior.
        self.pooling_layer(2, 2)
        
        #Terceira camada de convolução
        self.add_layer(32, 3, 3)

        #Aplicação a camada de pooling à saída da camada de convolução anterior.
        self.pooling_layer(2, 2)

        '''
        Aplicamos o achatamento ou apenas Flatter para converter a estrutura de dados 2D
        resultado da camada anterior em uma estrutra 1D, ou seja, um vetor.
        '''
        # Passo 3 - Flattening
        #self.classifier.add(Flatten())
        self.flatter_function()
        
    def connect_layers(self):      
        # Passo 4 - Full connection
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
              
    # Treinando, compilando a rede neural convolucional
    # Pré-Processamento        
    def rnc_compile(self):
        '''
        - Compiilando a rede neural.
        - Será utilizado o otimizador "Adam", um excelente algoritmo de primeira ordem para 
        otimização baseada em gradiente de funções objetivas estocásticas, que toma como 
        base um estimativa adaptada de momentos de baixa ordem.
        - Função log loss com "Entropia binária cruzada", ela funciona bem com funções sigmóides.
        - A metrica será a acurárica, pois essa é maior preocupação no treinamento deste tipo de modelo.
        '''

        # Compilando a rede - rede construída
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Treinando a rede neural convolucional
        # Pré-Processamento
        '''
        - Parte de pré processamento nos dados, no caso as imagens.
        - Será utilizado a função ImageDataGenerator() do Keras, ajustando escala e zoom
        das imagens de treino e a escala das imagens de validação.
        - O pré processamento do dados é etapa crucial em um projeto de Machine Learning.
        '''
        # Criando os objetos train_datagen e validation_datagen com as regras de 
        # pré processamento das imagens
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)

        validation_datagen = ImageDataGenerator(rescale = 1./255)

        self.training_set = train_datagen.flow_from_directory(self._folder_dataset_training,
                                                         target_size = (256, 256),
                                                         batch_size = 32,
                                                         class_mode = 'binary')

        self.validation_set = validation_datagen.flow_from_directory(self._folder_dataset_validation,
                                                                target_size = (256, 256),
                                                                batch_size = 32,
                                                                class_mode = 'binary')
        
        self.rnc_training(self.validation_set, self.training_set)
                

    def rnc_training(self, validation_set, training_set):
        self.classifier.fit_generator(self.training_set,
                                      steps_per_epoch = self._steps_per_epoch,
                                      epochs = self._qtd_epoch,
                                      validation_data = self.validation_set,
                                      validation_steps = self._steps_validation)  
        
    def rnc_process(self):
        self.build_rnc()
        self.connect_layers()
        self.rnc_compile()

    def rnc_prevision(self, image_teste):
        test_image = image.load_img(image_teste, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.classifier.predict(test_image)
        self.training_set.class_indices
        
        print(self.training_set.class_indices)
        print(result)

        if result[0][0] == 0:
            prediction = 'Cachorro'
      #  if result[0][0] == 2:
    #        prediction = 'Coelho'    
        else:
            prediction = 'Coelho'    
                    
        print(prediction)
        
        
rnc_objeto = RedesNeuraisConvolucionais(8000, 20, 2000, 'rnn_dataset_validation', 'rnn_dataset_treino')
rnc_objeto.rnc_process()
rnc_objeto.rnc_prevision('rnn_dataset_teste/8.jpg')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        