import tensorflow as tf 
from MRFN import MRFN
import os
import get_data
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import keras
import pandas as pd

import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import Input,Dense,RepeatVector,BatchNormalization,Activation,Conv1D,MaxPooling2D,UpSampling1D,Flatten,Conv2D,UpSampling2D
from keras.optimizers import  Adam
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import regularizers
from keras.utils import to_categorical



os.environ['PYTHONHASHSEED']=str(0)

class FitnessAssignment():
    def __init__(self, pops, params):
        self.pops = pops
        self.params = params

    def evalue_all(self, gen_no):#The scores of various groups in the gen_no iteration
        for i in range(self.pops.get_pop_size()):
            #tf.reset_default_graph()     
            mrfn = self.pops.indi[i]#Get the CAE parameters of the i-th population
            score = self.build_MRFN(mrfn)#获取
            #score = np.random.random()
            mrfn.score = score
#             list_save_path = os.getcwd() + '/save_data/pop_{:03d}.txt'.format(gen_no)
#             save_append_individual(str(cae), list_save_path)
            print('gen:{}, mrfn:{}/{}, accuracy:{:.6f}'.format(gen_no, i, self.pops.get_pop_size(), score))

    def build_MRFN(self, mrfn):
        hidden_layer=[]
        hidden_layer_flatten=[]
        input_img=Input(shape=(32, 32,1))
        for i in range(mrfn.num_channel):
            encoder=input_img
            for j in range(mrfn.num_stack):
                unit=mrfn.units[j+(i)*(mrfn.num_stack)]
                encoder= Conv2D(filters = unit.feature_size, kernel_size =unit.kernel, activation='relu', padding='same')(encoder)
                encoder= MaxPooling2D(pool_size = (2, 2), strides=(2,2),padding='same')(encoder)
            
            hidden_layer.append(encoder)
            flatten=Flatten()(encoder)
            hidden_layer_flatten.append(flatten)
        
        output=[]
        for i in range(mrfn.num_channel):
            decoder=hidden_layer[i]
            unit_feature_size=mrfn.units[(i+1)*mrfn.num_stack-1]
            decoder= Conv2D(filters = unit_feature_size.feature_size, kernel_size =3, activation='relu', padding='same')(decoder)
            for j in range(mrfn.num_stack):
                unit_kernel=mrfn.units[(mrfn.num_stack-1-j)+(i)*(mrfn.num_stack)]
                if j < mrfn.num_stack-1:
                    unit_feature_size=mrfn.units[(mrfn.num_stack-2-j)+(i)*(mrfn.num_stack)]
                    decoder= UpSampling2D((2, 2))(decoder) 
                    decoder= Conv2D(filters = unit_feature_size.feature_size, kernel_size =unit_kernel.kernel, activation='relu', padding='same')(decoder)                
                if j==mrfn.num_stack-1:
                    decoder= UpSampling2D((2, 2))(decoder) 
                    decoder= Conv2D(filters = 1, kernel_size =unit_kernel.kernel, activation='relu', padding='same')(decoder) 
           
            output.append(decoder)

        if mrfn.num_channel==1:
            flatten=np.array(hidden_layer_flatten)
        if mrfn.num_channel>1:
            flatten=keras.layers.concatenate(hidden_layer_flatten,axis=1)
        
        dense=Dense(256,activation="relu")(flatten)

        LR=Dense(8,activation="softmax",name='LR')(dense)

        output.append(LR)

        autoencoder=Model(inputs=input_img,outputs=output)
        
        autoencoder.summary()
        
        loss_name=[]
        for i in range(mrfn.num_channel):
            loss_name.append('mean_squared_error')
        loss_name.append('categorical_crossentropy')
        
        loss_weight=[]
        for i in range(mrfn.num_channel):
            loss_weight.append(round(0.1/(mrfn.num_channel),2))
        loss_weight.append(1)
        
        autoencoder.compile(metrics=['accuracy'],
                    loss=loss_name,
                    optimizer=Adam(lr=0.001),
                    loss_weights=loss_weight)
        
        X_train=self.params['X_train']
        y_train=self.params['y_train']
        X_test=self.params['X_test']
        y_test=self.params['y_test']
        train_data=[]
        test_data=[]
        
        for i in range(mrfn.num_channel):
            train_data.append(X_train)
            test_data.append(X_test)
        train_data.append(y_train)
        test_data.append(y_test)

        history = autoencoder.fit(X_train,
                          train_data,
                          validation_data=(X_test,test_data),
                          epochs= self.params['epochs'],
                          batch_size= self.params['batch_size'],
                          shuffle=True,
                          verbose=1).history
        accuracy=history['val_LR_accuracy']
        return max(accuracy)

if __name__ == '__main__':
    X_train=get_data.get_X_train()
    y_train=get_data.get_y_test()
    X_test=get_data.get_X_test()
    y_test=get_data.get_y_test()

    params = {}
    params['X_train'] = X_train
    params['y_train'] = y_train
    params['X_test'] = X_test
    params['y_test'] = y_test
    params['max_channel']=5
    params['max_stack']=5
    params['pop_size'] = 50
    params['num_class'] = 8
    params['total_generation'] = 50

    params['batch_size'] = 32
    params['epochs'] = 20


    mrfn = MRFN()
    mrfn_conv = mrfn.init(5,5)

    f= FitnessAssignment(None, params)
    f.build_MRFN(mrfn_conv)



