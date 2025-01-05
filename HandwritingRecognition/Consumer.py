import json
from channels.generic.websocket import AsyncWebsocketConsumer

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import re
import base64
from io import BytesIO
from PIL import Image

def predict(model, image):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(image)
    image = cv2.resize(image, (28, 28))
    #print(image)
    image = image / 255.0
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    prediction = model.predict(image)
    best_predictions = dict()

    for i in range(3):
        max_i = np.argmax(prediction[0])
        acc = round(prediction[0][max_i], 1)
        if acc > 0:
            label = labels[max_i]
            best_predictions[label] = acc
            prediction[0][max_i] = 0
        else:
            break

    return best_predictions

def load_model(path):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(36, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights(path)

    return model

model = load_model(r"C:\Users\debraj\Desktop\Demo_Project\HandwritingRecognition\best_val_loss_model.h5")
class HandWriting(AsyncWebsocketConsumer):
    #model = load_model(r"C:\Users\debraj\Desktop\Demo_Project\HandwritingRecognition\best_val_loss_model.h5")
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        await self.close()

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        #print(text_data_json)
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        expression = text_data_json['expression']
        ImageData = dataUrlPattern.match(expression).group(2)
        if (ImageData == None or len(ImageData) == 0):
            result = "Please Draw Characters"
            await self.send(text_data=json.dumps({
                'result': result
            }))
        else:
            ImageData = base64.b64decode(ImageData)
            ImageData = BytesIO(ImageData)
            im = Image.open(ImageData)
            image = np.array(im)
            prediction = predict(model, image)
            for i in prediction:
                #print(prediction[i])
                prediction[i]=int(prediction[i]*100)
            await self.send(text_data=json.dumps({
                'result': str(prediction)
            }))

