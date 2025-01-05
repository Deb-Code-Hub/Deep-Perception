import json
from channels.generic.websocket import AsyncWebsocketConsumer
import re
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

net = cv2.dnn.readNet(r"C:\Users\debraj\Desktop\Demo_Project\ObjectDetection\yolov3-tiny.weights",
                          r"C:\Users\debraj\Desktop\Demo_Project\ObjectDetection\yolov3-tiny.cfg.txt")
#print(net)
classes = []
with open(r"C:\Users\debraj\Desktop\Demo_Project\ObjectDetection\coco.names.txt", "rt") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = ['yolo_16', 'yolo_23']#[layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN
class ObjDec(AsyncWebsocketConsumer):

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

            height, width, channels = rgb_framee.shape
            # Detecting objects
            blob = cv2.dnn.blobFromImage(rgb_framee, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(rgb_framee, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(rgb_framee, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(rgb_framee, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3,
                                (255, 255, 255), 3)



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
