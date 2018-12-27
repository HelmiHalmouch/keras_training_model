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
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# import minivggnet from the loac folder 
from minivggnet_model.minivggnet import MiniVGGNet

'''in cmd use this command to get main idea about the training in keras :
						help(keras.engine.training)'''

#------------------General keras training function----------------------#
	#-1-# fit 			: model.fit(trainX, trainY, batch_size=32, epochs=50)
	#-2-# fit_generator :  model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	#												validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	#												epochs=EPOCHS)
	#-3-# train_on_batch: model.train_on_batch(batchX, batchY)

#------------------------Preprocessing of datasets------------------------#

#Function to load CSV imagedata file and loading images into memory
#Each line of data in the CSV file contains an image serialized as a text string.
def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
	'''
    inputPath : the path to the CSV dataset file.
    bs : The batch size. We’ll be using 32.
    lb : A label binarizer object which contains our class labels.
    mode : (default is "train" ) If and only if the mode=="eval" , then a special accommodation is made to not apply data augmentation via the aug  object (if one is supplied).
    aug : (default is None ) If an augmentation object is specified, then we’ll apply it before we yield our images and labels.

	'''
	# open the CSV file for reading
	f = open(inputPath, "r")
		# loop indefinitely
	while True:
		# initialize our batches of images and labels
		images = []
		labels = []

		# keep looping until we reach our batch size
		while len(images) < bs:
			# attempt to read the next line of the CSV file
			line = f.readline()

			# check to see if the line is empty, indicating we have
			# reached the end of the file
			if line == "":
				# reset the file pointer to the beginning of the file
				# and re-read the line
				f.seek(0)
				line = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the label and construct the image
			line = line.strip().split(",")
			label = line[0]
			image = np.array([int(x) for x in line[1:]], dtype="uint8")
			image = image.reshape((64, 64, 3))

			# update our corresponding batches lists
			images.append(image)
			labels.append(label)

		# one-hot encode the labels
		labels = lb.transform(np.array(labels))

		# if the data augmentation object is not None, apply it
		if aug is not None:
			(images, labels) = next(aug.flow(np.array(images),
				labels, batch_size=bs))

		# yield the batch to the calling function
		yield (np.array(images), labels)

#----------------------------Initialize our training parameters---------------------#
# initialize the paths to our training and testing CSV files
TRAIN_CSV = "flowers17_training.csv"
TEST_CSV = "flowers17_testing.csv"
 
# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 75
BS = 32
 
# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

How to use Keras fit and fit_generator (a hands-on tutorial)
Python
# open the training CSV file, then initialize the unique set of class
# labels in the dataset along with the testing labels
f = open(TRAIN_CSV, "r")
labels = set()
testLabels = []

# loop over all rows of the CSV file
for line in f:
	# extract the class label, update the labels list, and increment
	# the total number of training images
	label = line.strip().split(",")[0]
	labels.add(label)
	NUM_TRAIN_IMAGES += 1

# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")

# loop over the lines in the testing file
for line in f:
	# extract the class label, update the test labels list, and
	# increment the total number of testing images
	label = line.strip().split(",")[0]
	testLabels.append(label)
	NUM_TEST_IMAGES += 1

# close the testing CSV file
f.close()
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
	
# open the training CSV file, then initialize the unique set of class
# labels in the dataset along with the testing labels
f = open(TRAIN_CSV, "r")
labels = set()
testLabels = []
 
# loop over all rows of the CSV file
for line in f:
	# extract the class label, update the labels list, and increment
	# the total number of training images
	label = line.strip().split(",")[0]
	labels.add(label)
	NUM_TRAIN_IMAGES += 1
 
# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")
 
# loop over the lines in the testing file
for line in f:
	# extract the class label, update the test labels list, and
	# increment the total number of testing images
	label = line.strip().split(",")[0]
	testLabels.append(label)
	NUM_TEST_IMAGES += 1
 
# close the testing CSV file
f.close()




#my_model.summary()
if __name__ == '__main__':
	print('processing finished !!')