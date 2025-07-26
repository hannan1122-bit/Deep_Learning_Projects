import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# print(tf.__version__)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test / 255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model=Sequential(
    [
        Flatten(input_shape=(28,28)),
        Dense(128,activation='relu'),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')
    ]
)

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=32,verbose=1)
loss,accuracy=model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
model.save('Hand_Written_Text_Classifier_model.h5')

