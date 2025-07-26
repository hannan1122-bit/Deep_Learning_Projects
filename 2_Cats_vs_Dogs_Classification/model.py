import tensorflow as tf
import os
import zipfile

# -----------------------SECTION 1 DOWNLOADING DATA AND TAKING ITS PATH--------------------------

# THIS SECTION NEEDS ONLY ONE TIME TO RUN BECAUSE WE DOWNLOAD DATA ONLY ONE ITME
# URL="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
# zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip',origin=URL,extract =True)
# base_dir = os.path.join(os.path.dirname(zip_dir),'cats_and_dogs_filtered')


# MAKING DATA PATHS TO VALIDATION AND TRAINING
base_dir = "/home/abdul/.keras/datasets/cats_and_dogs_filtered_extracted/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')


# -----------------------------PREPROCESSING FOR IMAGES BEFORE GOING TO CNN MODEL-------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# DIFFERNT ORIENTATION BASED IMAGES GENERATED BECAUSE WE NEED VARIATION TO TRAIN MODEL BEST
train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=30,
                                 horizontal_flip=True,
                                 zoom_range=0.2,
                                 shear_range=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,

                                 )
validation_datagen=ImageDataGenerator(rescale=1./255)

# LOADING DATA FROM DIRECTORIES WITH SIZE OF 32 AND TAKING IMAGE SIZES BINARY 0-CAT 1-DOG
train_generator=train_datagen.flow_from_directory(
    train_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(250,250)
)
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(250, 250)
)


# ----------------------------------MODEL CREATTION-------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(250, 250, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ----------------------------------MODEL CREATTION-------------------------------

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

# DONT NEED TO GIVE BATCHES SIZE HERE BECAUSE IT COMES FROM GENERATORS
history=model.fit(train_generator,epochs=20,validation_data=validation_generator)
print("NORMAL ACCURACY: ",history.history['accuracy'])
print("VALIDATION ACCURACY",history.history['val_accuracy'])

loss,accuracy=model.evaluate(validation_generator)
print("Loss:", loss)
print("Accuracy:", accuracy)

model.save("cat_dog_cnn_model.h5")
