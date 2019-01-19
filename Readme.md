# 1-Main objective 
Application of the following keras methods for the training of a model :<br/>
a-fit <br/>
b-fit_generator <br/>
c-train_on_batch <br/>

# 2-General keras training function

-1- fit 			: model.fit(trainX, trainY, batch_size=32, epochs=50) <br/>
-2- fit_generator   : model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_						data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS) <br/> 
-3- train_on_batch  : model.train_on_batch(batchX, batchY) <br/>

## How choose the training keras function ?
a. keras .fit function is acceptable for a small dataset <br/>
b. keras .fit_generator is used when we need to perform the dataset augmentation <br/>
c. keras .train_on_batch  is applied for the finest-grained control over training your Keras models<br/>

## Requirement 
Python3.x > 3.4 <br/>
TensorFlow + Keras <br/>
Scikit-learn <br/>
NumPy <br/>
Matplotlib <br/>

## Used datasets 
The datasets used in our code can be download from this link: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
<br/>
the datasets in csv file are : flowers17_testing.csv  and  flowers17_training.csv
<br/>
To run the code, you can use : python keras_fit_fitgenerator.py

## Used architecture model (minivggnet)
![minivggnet](https://user-images.githubusercontent.com/40611217/50492086-e2f7ea80-0a15-11e9-9f7f-2a09f57bbc23.JPG)


