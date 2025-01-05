import json
from channels.generic.websocket import AsyncWebsocketConsumer
import re
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model(r"C:\Users\debraj\Desktop\Demo_Project\SignLanguageDetection\mp_hand_gesture")

# Load class names
f = open(r"C:\Users\debraj\Desktop\Demo_Project\SignLanguageDetection\gesture.names", 'rt')
classNames = f.read().split('\n')
f.close()
font = cv2.FONT_HERSHEY_PLAIN

class SignDec(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        await self.close()

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        # print(text_data_json)
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        expression = text_data_json['expression']
        ImageData = dataUrlPattern.match(expression).group(2)
        # print(type(ImageData))#<class 'str'>

        if (ImageData == None or len(ImageData) == 0):
            result = "Please show objects to the camera"
            await self.send(text_data=json.dumps({
                'result': result
            }))
        else:

            ImageData = base64.b64decode(ImageData)
            # print(ImageData)
            # print(type(ImageData))#<class 'bytes'>
            ImageData = BytesIO(ImageData)
            # print(ImageData)
            # print(type(ImageData))#<class '_io.BytesIO'>
            imageFrame = Image.open(ImageData)
            # print(type(imageFrame))#<class 'PIL.PngImagePlugin.PngImageFile'>

            # imag=cv2.imread(str(ImageData))
            # imag = np.array(imageFrame)
            # rgb_frame = imageFrame[:, :, ::-1]
            # rgb_frame=Image.fromarray(imag,"RGB")

            rgb_fram = imageFrame.convert('RGB')

            # print(type(rgb_fram)) #<class 'PIL.Image.Image'>
            # x = rgb_fram.save("In.jpg")

            rgb_framee = np.array(rgb_fram)

            # rgb_frame = rgb_framee[:, :, ::-1]
            #x, y, c = imageFrame.shape

            # Flip the frame vertically
            #f_rame = cv2.flip(imageFrame, 1)
            #framergb = cv2.cvtColor(f_rame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            #result = VideoCamera.hands.process(framergb)

            x, y, c = rgb_framee.shape

            # Flip the frame vertically
            #f_rame = cv2.flip(rgb_framee, 1)
            #framergb = cv2.cvtColor(rgb_framee, cv2.COLOR_RGB2BGR)
            #framergb=cv2.cvtColor(framergb, cv2.COLOR_RGB2BGR)

            # Get hand landmark prediction
            res = hands.process(rgb_framee)

            # print(result)

            className = ''

            # post process the result
            if res.multi_hand_landmarks:
                landmarks = []
                for handslms in res.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(rgb_framee, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = model.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    className =classNames[classID]

            # show the prediction on the frame
            cv2.putText(rgb_framee, className, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 2, cv2.LINE_AA)


            # imaget=rgb_fram.convert
            # x = Image.fromarray(rgb_framee)
            # print(type(x))
            # x=Image.open(x)
            # print(type(x))
            # x=x.tobytes()
            # result=rgb_framee.tobytes()
            # result=base64.b64encode(result)
            # x=BytesIO(x)
            # ae=x.save('out4.jpg')
            # print(ae)
            # x=rgb_framee.tobytes()
            # with open("out2.jpg", "rb") as image_file:
            # result = base64.b64encode(image_file.read()).decode()
            # print(result)
            # result=cv2.imread("out2.jpg")
            # result = base64.b64encode(result).decode()
            # print(result)
            # print(type(result))
            # result=base64.b64encode(image_file.read()).decode()
            # result="iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACN" \
            # "byblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

            # result=result.decode()
            # print(result)
            # print(type(result))
            # elapsed_time = time.time() - .starting_time
            # fps = frame_id / elapsed_time
            # cv2.putText(imageFrame, "FPS: " + str(round(fps, 2)), (380, 40), VideoCamera.font, 3, (0, 50, 100), 3)

            # image = np.array(im)
            # prediction = predict(model, image)
            # for i in prediction:
            # print(prediction[i])
            # prediction[i]=int(prediction[i]*100)
            # print(rgb_framee)
            # result=rgb_framee.tobytes()
            x = Image.fromarray(rgb_framee)
            s = BytesIO();
            x.save(s, "png")
            result = base64.b64encode(s.getvalue()).decode()
            await self.send(text_data=json.dumps({
                'result': str(result)
            }))
