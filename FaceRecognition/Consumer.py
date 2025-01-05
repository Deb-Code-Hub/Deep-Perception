import json
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import numpy as np
import face_recognition
import re
import base64
from io import BytesIO
from PIL import Image

color1 = list(np.random.choice(range(255), size=3))
color = list(np.random.choice(range(255), size=3))
# Load first sample picture and learn how to recognize it.
Debraj_image = face_recognition.load_image_file(
    r"C:\Users\debraj\Desktop\Demo_Project\FaceRecognition\Debraj.jpg")
Debraj_face_encoding = face_recognition.face_encodings(Debraj_image)[0]

# Load a second sample picture and learn how to recognize it.
Damodar_image = face_recognition.load_image_file(
    r"C:\Users\debraj\Desktop\Demo_Project\FaceRecognition\Damodar.jpg")
Damodar_face_encoding = face_recognition.face_encodings(Damodar_image)[0]

# Load a third sample picture and learn how to recognize it.
Renish_image = face_recognition.load_image_file(
    r"C:\Users\debraj\Desktop\Demo_Project\FaceRecognition\Renish.jpg")
Renish_face_encoding = face_recognition.face_encodings(Renish_image)[0]

# Load a fourth sample picture and learn how to recognize it.
Chiranjit_image = face_recognition.load_image_file(
    r"C:\Users\debraj\Desktop\Demo_Project\FaceRecognition\Chiranjit.jpg")
Chiranjit_face_encoding = face_recognition.face_encodings(Chiranjit_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Debraj_face_encoding,
    Damodar_face_encoding,
    Renish_face_encoding,
    Chiranjit_face_encoding

]
known_face_names = [
    "Debraj",
    "Damodar",
    "Renish",
    "Chiranjit"

]
font = cv2.FONT_HERSHEY_PLAIN

class FaceRecog(AsyncWebsocketConsumer):

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
        #print(type(ImageData))#<class 'str'>

        if (ImageData == None or len(ImageData) == 0):
            result = "Please show your face to the camera"
            await self.send(text_data=json.dumps({
                'result': result
            }))
        else:

            ImageData = base64.b64decode(ImageData)
            #print(ImageData)
            #print(type(ImageData))#<class 'bytes'>
            ImageData = BytesIO(ImageData)
            #print(ImageData)
            #print(type(ImageData))#<class '_io.BytesIO'>
            imageFrame = Image.open(ImageData)
            #print(type(imageFrame))#<class 'PIL.PngImagePlugin.PngImageFile'>

            #imag=cv2.imread(str(ImageData))
            #imag = np.array(imageFrame)
            #rgb_frame = imageFrame[:, :, ::-1]
            #rgb_frame=Image.fromarray(imag,"RGB")
            rgb_fram=imageFrame.convert('RGB')
            #print(type(rgb_fram)) #<class 'PIL.Image.Image'>
            #x = rgb_fram.save("In.jpg")
            rgb_framee=np.array(rgb_fram)
            #rgb_frame = rgb_framee[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_framee)#rgb_frame
            face_encodings = face_recognition.face_encodings(rgb_framee, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Cannot Recognize"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                cv2.rectangle(rgb_framee, (left, top - 35), (right, bottom),
                              (int(color[0]), int(color[1]), int(color[2])), 2)

                cv2.putText(rgb_framee, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                            (int(color1[0]), int(color1[1]), int(color1[2])), 1)
            #imaget=rgb_fram.convert
            #x = Image.fromarray(rgb_framee)
            #print(type(x))
            #x=Image.open(x)
            #print(type(x))
            #x=x.tobytes()
            #result=rgb_framee.tobytes()
            #result=base64.b64encode(result)
            #x=BytesIO(x)
            #ae=x.save('out4.jpg')
            #print(ae)
            #x=rgb_framee.tobytes()
            #with open("out2.jpg", "rb") as image_file:
                #result = base64.b64encode(image_file.read()).decode()
            #print(result)
            #result=cv2.imread("out2.jpg")
            #result = base64.b64encode(result).decode()
            #print(result)
            #print(type(result))
            #result=base64.b64encode(image_file.read()).decode()
            #result="iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACN" \
                   #"byblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

            #result=result.decode()
            #print(result)
            #print(type(result))
            #elapsed_time = time.time() - .starting_time
            #fps = frame_id / elapsed_time
            #cv2.putText(imageFrame, "FPS: " + str(round(fps, 2)), (380, 40), VideoCamera.font, 3, (0, 50, 100), 3)

            #image = np.array(im)
            #prediction = predict(model, image)
            #for i in prediction:
                #print(prediction[i])
                #prediction[i]=int(prediction[i]*100)
            #print(rgb_framee)
            #result=rgb_framee.tobytes()
            x = Image.fromarray(rgb_framee)
            s=BytesIO();
            x.save(s, "png")
            result = base64.b64encode(s.getvalue()).decode()
            await self.send(text_data=json.dumps({
                'result': str(result)
            }))
