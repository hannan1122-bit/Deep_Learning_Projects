import cv2
import numpy as np
from tensorflow.keras.models import load_model

img=cv2.imread("cat.png")
img=cv2.resize(img,(160,160))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=img/255.0
img=np.expand_dims(img, axis=0)

model=load_model("cat_dog_transfer_model.h5")

prediction=model.predict(img)

if prediction[0][0]>0.5:
    print("It's a dog")
else:
    print("It's a cat")