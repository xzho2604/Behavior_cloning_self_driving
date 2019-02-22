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

    
    img = image.load_img(curr_path,target_size = (32,32))
    img = image.img_to_array(img)
#   image = ndimage.imread(curr_path) 
    images.append(img)
    mesure = float(line[3]) #extract the 4th column control 
    mesures.append(mesure)



#defin training data and model
images = np.array(images)
mesures = np.array(mesures)
print(images.shape)
images = np.subtract(np.true_divide(images, 225.0) ,0.5)

X_train = images
y_train = mesures

from keras.models import Sequential ,Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Lambda,Input
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

inception = InceptionV3(weights='imagenet', include_top=False,input_shape = (160,320,3))

#freeze all the layers
for layer in inception.layers:
    layer.trainable = False

#resize the input image to 32,32,3
cifar_input = Input(shape=(32,32,3))
resized_input = Lambda(lambda image: tf.image.resize_images(image, (160, 320)))(cifar_input)
inp = inception(resized_input)

x = GlobalAveragePooling2D()(inp)
x = Dense(128, activation = 'relu')(x)
x = Dense(32)(x)
predictions = Dense(1)(x)

model = Model(inputs=cifar_input, outputs=predictions)
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) #crop the input image
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #normalise the input images
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,verbose=1,epochs = 10)

model.save('model.h5')









