from tensorflow.keras import layers,models,optimizers
import matplotlib.pyplot as plt
import numpy as np

print('loading data')
y_train=np.load('y_train.npy')
x_train=np.load('x_train.npy')
y_test=np.load('y_test.npy')
x_test=np.load('x_test.npy')

epoch=5
inputShape=x_train[0].shape
batchSize=x_train.shape[0]//50

shapeClassifier=models.Sequential([
    layers.Conv2D(filters=50,kernel_size=(4,4),activation='relu',
                  input_shape=inputShape),
    layers.MaxPooling2D(pool_size=(4, 4)),

    layers.Conv2D(filters=25,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3)),
    
    layers.Conv2D(filters=12,kernel_size=(2,2),activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3)),
    
    layers.Flatten(),
    #layers.Dense(units=16,activation='relu'),
    #layers.Dense(units=8,activation='relu'),
    layers.Dense(units=4,activation='relu'),
    layers.Dense(units=2,activation='softmax')])

shapeClassifier.compile(
    optimizer=optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999),
    loss='categorical_crossentropy',
    metrics='accuracy')

print('starting training')
history=shapeClassifier.fit(
    x_train,y_train[:,0:2],
    validation_data=(x_test,y_test[:,0:2]),
    epochs=epoch,
    batch_size=batchSize,
    verbose=1)

print('displaying stats')

figure, axis = plt.subplots(2)
axis[0].plot(history.history['accuracy'],color='blue')
axis[0].plot(history.history['val_accuracy'],color='yellow')
axis[0].set_title('acc vs epoch',loc='left')
axis[0].set_xlabel('Epoch')
axis[0].legend(['train set','test set'], loc='upper left')

y_predict=shapeClassifier.predict(x_test,verbose=0)
sample=np.arange(0,y_predict.shape[0])
predicted_shape=np.argmax(y_predict,axis=1)
actual_shape=np.argmax(y_test[:,0:2],axis=1)
axis[1].plot(sample,actual_shape,color='red',marker='s')
axis[1].plot(sample,predicted_shape,color='blue',marker='.')
axis[1].set_title('truth vs model',loc='left')
axis[1].set_xlabel('sample')
axis[1].legend(['truth','model'], loc='upper left')
plt.show()
saveConfirmation=input('save shapeClassifier? (y/n) ')
if saveConfirmation=='y' or 'Y':shapeClassifier.save('shapeClassifier')
