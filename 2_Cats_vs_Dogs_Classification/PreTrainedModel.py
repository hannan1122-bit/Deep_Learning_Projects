import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import os

# ------------------ SECTION 1: Paths ------------------
base_dir = "/home/abdul/.keras/datasets/cats_and_dogs_filtered_extracted/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ------------------ SECTION 2: Image Generators --------------
IMG_SIZE = (160, 160)  # smaller than 250x250 = faster training

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

validation_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

# ------------------ SECTION 3: Transfer Learning ------------------
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze weights for faster training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ------------------ SECTION 4: Compile Model ------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------ SECTION 5: Callbacks ------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ------------------ SECTION 6: Train ------------------
history = model.fit(
    train_generator,
    epochs=10,  # Less epochs due to fast convergence
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# ------------------ SECTION 7: Evaluate & Save ------------------
val_loss, val_acc = model.evaluate(validation_generator)
print("✅ Final Validation Accuracy:", round(val_acc * 100, 2), "%")

model.save("cat_dog_transfer_model.h5")
print("✅ Model saved as 'cat_dog_transfer_model.h5'")
