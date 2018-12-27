# -*- coding: utf-8 -*-

'''
Application of fit, fit_generator and train_on_batch methods 
									in keras for training of a model 

GHNAMI Helmi 
27/12/2018

'''

#import librery and packages 
import sys, os 
import numpy as np 
import matplotlib.pyplot as plt 
import keras 

'''in cmd use this command to get main idea about the training in keras :
						help(keras.engine.training)'''

#------------------General keras training function----------------------#
	#-1-# fit 			: model.fit(trainX, trainY, batch_size=32, epochs=50)
	#-2-# fit_generator :  model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	#												validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	#												epochs=EPOCHS)
	#-3-# train_on_batch: model.train_on_batch(batchX, batchY)

if __name__ == '__main__':
	print('processing finished !!')