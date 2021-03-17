
# import libraries

from tensorflow import keras
from datetime import datetime

from src.support.evaluation import *


from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model

# set model name

model_name = "mobileNet_anomaly_detection_v1"

# define dataset directories

classes = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus',
           'normal-z-line', 'polyps', 'ulcerative-colitis']
root_dir = '../../data/'

# define ImageGenarator and data augmentation

train_gen_tmp = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)

train_gen = train_gen_tmp.flow_from_directory(root_dir,
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              batch_size=20,
                                              shuffle=True,
                                              seed=42,
                                              subset='training')

validation_gen = train_gen_tmp.flow_from_directory(root_dir,
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   class_mode='categorical',
                                                   batch_size=20,
                                                   shuffle=True,
                                                   seed=42,
                                                   subset='validation')

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = validation_gen.n // validation_gen.batch_size

clToInt_dict = train_gen.class_indices
clToInt_dict = dict((k, v) for v, k in clToInt_dict.items())

# define the model

weight_path = '../../h5_files/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'

pre_model = MobileNetV2(input_shape=(224,224,3),
                    weights = None,
                    include_top = True)

pre_model.load_weights(weight_path)

for layer in pre_model.layers:
    layer.trainable = False

conn_layer = pre_model.get_layer('block_12_add')
conn_output = conn_layer.output

x = Conv2D(128,(3,3),activation='relu')(conn_output)
x = MaxPool2D(2,2)(x)
x = Conv2D(256,(3,3),activation='relu')(conn_output)
x = MaxPool2D(2,2)(x)
x = Flatten()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128,activation='relu')(x)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Dense(8,activation='softmax')(x)

model = Model(pre_model.input,x)

#model.summary()

# set model optimizer and loss function

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=15,
                    verbose=1)

# save the model

model.save('../../h5_files/final_model.h5')

# summary of training results

acc_n_loss(history)
