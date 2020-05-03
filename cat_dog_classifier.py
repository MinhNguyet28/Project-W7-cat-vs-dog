# Import libaries
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os 
import cv2
import time
import sys

from os import makedirs, listdir
from shutil import copyfile,move
from random import seed
from random import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#Helper 
def doesDataSetContainsEnoughDataForBatch(dataset, batch_size):
    return len(list(dataset.take(batch_size).as_numpy_iterator())) == batch_size
  
def doesDataSetFileContainsEnoughDataForBatch(sampleFileName="", batch_size=100):
    dataset = tf.data.TFRecordDataset(sampleFileName)
    return doesDataSetContainsEnoughDataForBatch(dataset, batch_size=batch_size)

def removeipynbCheckpoints(path):
    try:
      os.remove(os.path.join(path,'.ipynb_checkpoints'))
    except:
      print('{} ipynb not exist'.format(path))
    
def prepareSubClassFolder(dir):
    dogs = os.path.join(dir, 'dogs')
    cats = os.path.join(dir, 'cats')
    makedirs(dogs, exist_ok=True)
    makedirs(cats, exist_ok=True)

def subClassHelper(dir):
    prepareSubClassFolder(dir)
    for file in listdir(dir):
      src = os.path.join(dir,file)
      dst = ''
      if file.startswith('cat'):
        dst = os.path.join(dir,'cats',file) 
      elif file.startswith('dog'):
        dst = os.path.join(dir,'dogs',file) 
      if(os.path.isdir(src)): 
        continue
      move(src, dst)
# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('./train_split/_plot.png')
    plt.close()


# Create directories
dataset_home = './train_split/'
subdirs = ['train_i/', 'val_i/', 'test_i/']

# create label subdirectories
for subdir in subdirs:
  newdir = dataset_home + subdir 
  makedirs(newdir, exist_ok=True)
  removeipynbCheckpoints(newdir)

seed(1)

src_directory = './train'
train_i_dir = './train_split/train_i/'
val_i_dir = './train_split/val_i/'
test_i_dir = './train_split/test_i/'

#Total list image
lst_img = []
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if(os.path.isdir(src)):
		continue
	else:
		lst_img.append(src)

#shuffle x2 
shuffle(lst_img)
shuffle(lst_img)


#define configs
BATCH_SIZE = 64
IMAGE_SIZE = 224
DATASET_SIZE = len(lst_img)
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

count_train = 0
count_val = 0
count_test = 0
for img in range(len(lst_img)):
	fileName = os.path.basename(lst_img[img])
	if img < train_size:
		copyfile(lst_img[img], train_i_dir + fileName)
		count_train += 1
	elif img < (train_size + val_size):
		copyfile(lst_img[img], val_i_dir + fileName)
		count_val += 1
	else:
		copyfile(lst_img[img], test_i_dir + fileName)
		count_test +=1

#log folder data size
print('train_i {}, val_i {}, test_i {}'.format(count_train,count_val,count_test))

#Subclass train_i
subClassHelper(train_i_dir)
#Subclass val_i
subClassHelper(val_i_dir)
#Subclass test_i
subClassHelper(test_i_dir)

# this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     horizontal_flip=True)
# val_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
train_datagen.mean = [123.68, 116.779, 103.939]

train_generator = train_datagen.flow_from_directory(
        train_i_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE), 
        batch_size = BATCH_SIZE, 
        class_mode='binary')

# validation_generator = val_datagen.flow_from_directory(
#         val_i_dir,
#         target_size=(IMAGE_SIZE, IMAGE_SIZE), 
#         batch_size = BATCH_SIZE, 
#         class_mode='binary')

# Define model
def define_model():
    # load model
    model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
    # model = tf.keras.applications.Xception(include_top=False, input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers VGG16
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # add new classifier layers Xception
    # x = model.output
    # x = GlobalAveragePooling2D(name='avg_pool')(x) # you don't have to name the layer
    # x = Dropout(0.25)(x) # based on the tunned hyperparamters of model 3
    # x = Dense(256, activation = 'relu')(x)
    # output = Dense(1, activation='sigmoid')(x)

    # define new model
    model = tf.keras.Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    # opt = tf.keras.optimizers.RMSprop(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

print(model.summary())
# Train model
model_history = model.fit_generator(
    train_generator,
    steps_per_epoch= len(train_generator),
    epochs = 10,
    # validation_data = validation_generator,
    # validation_steps= count_val // BATCH_SIZE,
    verbose = 1)

# Evaluate model
# model.evaluate_generator(validation_generator, (count_val // BATCH_SIZE))
model.save('./catdog.h5')

# summarize_diagnostics(model_history)

# Save model

