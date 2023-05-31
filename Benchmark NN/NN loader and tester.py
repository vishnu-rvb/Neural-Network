from tensorflow.keras import layers,models,optimizers
import matplotlib.pyplot as plt
import numpy as np

print('loading data')

y_test=np.load('y_test.npy')
x_test=np.load('x_test.npy')
shapeClassifier=models.load_model('shapeClassifier')

print('displaying stats')

y_predict=shapeClassifier.predict(x_test,verbose=0)
sample=np.arange(0,y_predict.shape[0])
predicted_shape=np.argmax(y_predict,axis=1)
actual_shape=np.argmax(y_test[:,0:2],axis=1)
for i in sample:
    if actual_shape[i]!=predicted_shape[i]:
        print(i,actual_shape[i],predicted_shape[i])
#figure, axis = plt.subplots(2)
#axis[0].scatter(sample,actual_shape,color='red',marker='s')
#axis[0].scatter(sample,predicted_shape,color='blue',marker='.')
#axis[0].set_title('truth vs model',loc='left')
#axis[0].set_xlabel('sample')
#axis[0].legend(['truth','model'], loc='upper left')
#plt.show()
