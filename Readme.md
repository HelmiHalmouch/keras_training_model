# 1-Main objective 
Application of fit, fit_generator and train_on_batch methods <br/>
in keras for training of a model <br/>

# 2-General keras training function

-1- fit 			: model.fit(trainX, trainY, batch_size=32, epochs=50) <br/>
-2- fit_generator   : model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_						data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS) <br/> 
-3- train_on_batch  : model.train_on_batch(batchX, batchY) <br/>

## How choose the training keras function ?
a. keras .fit function is acceptable for a small dataset <br/>
b. keras .fit_generator is used when we need to perform the dataset augmentation <br/>
c. keras .train_on_batch  is applied for the finest-grained control over training your Keras models<br/>

