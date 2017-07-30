from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import numpy
import pandas
from pandas import Series

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

seed = 7
epoch=1000
#batchSize=20000
batchSizeCoff = .4

inputDim=50
dataSplit = 0.1
validationSplit = 0.2


early_stopping = EarlyStopping(monitor='val_loss', patience=epoch*.3)
# checkpoint
filepath="weights/improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint,early_stopping]

trainFile = 'numerai_datasets-dec3016/numerai_training_data.csv'

saveModelPrefix = "weights/numer-dec3016-"

# fix random seed for reproducibility
numpy.random.seed(seed)

# load the dataset using numpy
#dataset = numpy.loadtxt(trainFile, delimiter=",",skiprows=1)

# load data using panda
dataframe = pandas.read_csv(trainFile)
dataset = dataframe.values
dataset = dataset.astype('float32')

# split into train and test sets
test_size = int(len(dataset) * dataSplit)
train_size = len(dataset) - test_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# split into input (X) and output (Y) variables
#for train and test
X = train[:,0:inputDim]
#X  = preprocessing.normalize(X)
Y = train[0:,inputDim]
#for validation
Xt = test[:,0:inputDim]
#Xt  = preprocessing.normalize(Xt)
Yt = test[0:,inputDim]

batchSize = int(len(X) * batchSizeCoff)

# create model
model = Sequential()
model.add(Dense(int(inputDim*.9), input_dim=inputDim, init='he_uniform', activation='relu'))
model.add(Dense(int(inputDim*.9), init='he_uniform', activation='relu'))
model.add(Dense(int(inputDim*.9), init='he_uniform', activation='relu'))
#model.add(Dense(int(inputDim*.8), init='he_uniform', activation='relu'))

#model.add(Dense(inputDim, init='he_uniform', activation='sigmoid'))
#model.add(Dense(inputDim, init='he_uniform', activation='sigmoid'))
#model.add(Dense(inputDim, init='he_uniform', activation='sigmoid'))
model.add(Dense(1, init='he_uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=epoch, batch_size=batchSize,validation_split=validationSplit,callbacks=callbacks_list)

# evaluate the model
print 'evaluating model'
scores = model.evaluate(Xt, Yt)
print(" \n %s: %.4f" % (model.metrics_names[0], scores[0]))
print(" %s: %.4f%%" % (model.metrics_names[1], scores[1]*100))

nameSurfix = str(round(scores[0],4))+"-"+str(round(scores[1]*100,4))

# creates a HDF5 file
model.save(saveModelPrefix+nameSurfix+".h5")
