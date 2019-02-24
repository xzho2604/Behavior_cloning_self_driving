
import csv
from scipy import ndimage
import numpy as np
from keras.preprocessing import image

#open the file and load file into lines
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#load into images 
images = []
mesures = []
for line in lines:
    source_path = line[0] #get the centre camera image path(original)
    filename = source_path.split('/')[-1] #extract the only image name from the path
    curr_path = './data/IMG/' + filename  #concat as the current file path on aws

    
#    img = image.load_img(curr_path,target_size = (32,32))
#    img = image.img_to_array(img)
    img = ndimage.imread(curr_path) 
    images.append(img)
    mesure = float(line[3]) #extract the 4th column control 
    mesures.append(mesure)



#defin training data and model
images = np.array(images)
mesures = np.array(mesures)
print(images.shape)
#images = np.subtract(np.true_divide(images, 225.0) ,0.5)

X_train = images
y_train = mesures

from keras.models import Sequential ,Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D,Lambda,Input,Cropping2D,Conv2D, BatchNormalization,Activation
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

#the navida model
def create_navida():
    model = Sequential()

    # Normalization Layer
    #model.add(Lambda(lambda x: np.subtract(np.true_divide(x , 255.0) , 0.5), input_shape=(160,320,3)))
    #crop the data
    model.add(Cropping2D(cropping = ((50,20),(0,0)), input_shape = (160,320,3)))
    
    # Convolutional Layer 1
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 3
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 5
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flatten Layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1))

    # Configure learning process with an optimizer and loss function
    model.compile(loss='mse', optimizer='Adam',metrics=['accuracy'])

    return model

navida = create_navida()
print(navida.summary())
navida.fit(X_train, y_train, validation_split = 0.2, shuffle = True,verbose=1,epochs = 5)
navida.save('model.h5')


