# hello-deepworld
Iris classification with keras using tensorflow backend.
## Preprocessing of the packages:
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.callbacks import TensorBoard
import seaborn as sns

#Reads the data:
dataset = pd.read_csv("YOUR_CSV_PATH")


##Gets logs of the loss and accuracy functions:
tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


#### Plots the dataset by using seaborn:
##sns.set(style="ticks")
##sns.set_palette("husl")
##df = pd.DataFrame(data=dataset.iloc[:,1:6])
##sns.pairplot(df,hue="variety")
##plt.show()


## Seed the generator to re-seed the generator:
np.random.seed(5)


## Using data.iloc[<row selection>, <column selection>] syntax,
## to select row and column number:
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


## Encodes the y data with numbers between 0 and number of classes-1:
encoder = LabelEncoder()
y = encoder.fit_transform(y)


## Converts array of labeled data(from 0 to nb_classes-1) to one-hot vector.
y = np_utils.to_categorical(y)


## Splitting data to build test and train data.
## (80% training data & 20% test data) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


## StandardScaler transforms data to make scaled data has zero mean and unit variance: 
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train) # Compute mean, std and transform training data as well
x_test = scaler.transform(x_test)       # Perform standardization by centering and scaling


## Sequential API of Keras is used.
clf = Sequential()

## Hidden layer:
clf.add(Dense(kernel_initializer = 'uniform', input_dim = 4, units = 4, activation = 'relu'))

## Output layer:
clf.add(Dense(kernel_initializer = 'uniform', units = 3,  activation = 'softmax'))

## Configure the learning process, which is done via the compile method.
clf.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

## Train the model, iterating on the data in batches of 5 samples:
clf.fit(x_train, y_train, batch_size=5, epochs=100,callbacks=[tensor_board])


## Accuracy
y_pred = clf.predict(x_test)
y_test_class = np.argmax(y_test,axis=1) #Returns the indices of the maximum values along an axis.
y_pred_class = np.argmax(y_pred,axis=1)

print(accuracy_score(y_test_class, y_pred_class))


