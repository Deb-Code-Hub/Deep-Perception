import json
from channels.generic.websocket import AsyncWebsocketConsumer
import re
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

class ColourDec(AsyncWebsocketConsumer):

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
            result = "Please show some colours to the camera"
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
            #rgb_frame=Image.fromarray(imag,"RGB")
            #imageFrame=np.array(imageFrame)
            rgb_fram = imageFrame.convert('RGB')
            #cv2.imshow(rgb_fram)
            #imageFrame=cv2.cvtColor(imageFrame, cv2.COLOR_RGBA2BGR)

            # print(type(rgb_fram)) #<class 'PIL.Image.Image'>
            # x = rgb_fram.save("In.jpg")

            imageFrame = np.array(rgb_fram) #rgb frame

            # rgb_frame = rgb_framee[:, :, ::-1]
            #imageFrame=cv2.cvtColor(iimageFrame, cv2.COLOR_RGBA) #bgr frame

            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2HSV) #hsv frame

            #imageFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)

            # Set range for gray color and
            # define mask
            gray_lower = np.array([0, 0, 70], np.uint8)  # remember BGR
            gray_upper = np.array([170, 10, 170], np.uint8)
            gray_mask = cv2.inRange(hsvFrame, gray_lower, gray_upper)

            # Set range for black color and
            # define mask
            black_lower = np.array([0, 0, 0], np.uint8)  # remember BGR
            black_upper = np.array([0, 0, 255], np.uint8)
            black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

            # Set range for white color and
            # define mask
            white_lower = np.array([0, 0, 230], np.uint8)  # remember BGR
            white_upper = np.array([70, 0, 255], np.uint8)
            white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

            # Set range for yellow color and
            # define mask
            yellow_lower = np.array([30, 50, 175], np.uint8)  # remember BGR
            yellow_upper = np.array([34, 170, 255], np.uint8)
            yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

            # Set range for red color and
            # define mask
            red_lower = np.array([175, 85, 175], np.uint8)
            red_upper = np.array([180, 255, 205], np.uint8)
            red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

            # Set range for green color and
            # define mask
            green_lower = np.array([50, 105, 85], np.uint8)
            green_upper = np.array([85, 255, 255], np.uint8)
            green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

            # Set range for blue color and
            # define mask
            blue_lower = np.array([94, 120, 120], np.uint8)
            blue_upper = np.array([102, 255, 255], np.uint8)
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

            # Set range for violet color and
            # define mask
            violet_lower = np.array([130, 60, 111], np.uint8)
            violet_upper = np.array([140, 255, 255], np.uint8)
            violet_mask = cv2.inRange(hsvFrame, violet_lower, violet_upper)

            # Set range for orange color and
            # define mask
            orange_lower = np.array([12, 180, 230], np.uint8)
            orange_upper = np.array([20, 220, 255], np.uint8)
            orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

            # Set range for brown color and
            # define mask
            brown_lower = np.array([0, 100, 10], np.uint8)
            brown_upper = np.array([10, 220, 110], np.uint8)
            brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper)

            kernel = np.ones((5, 5), "uint8")

            # For gray color
            gray_mask = cv2.dilate(gray_mask, kernel)
            res_gray = cv2.bitwise_and(imageFrame, imageFrame,
                                       mask=gray_mask)

            # For black color
            black_mask = cv2.dilate(black_mask, kernel)
            res_black = cv2.bitwise_and(imageFrame, imageFrame,
                                        mask=black_mask)

            # For white color
            white_mask = cv2.dilate(white_mask, kernel)
            res_white = cv2.bitwise_and(imageFrame, imageFrame,
                                        mask=white_mask)

            # For yellow color
            yellow_mask = cv2.dilate(yellow_mask, kernel)
            res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                                         mask=yellow_mask)
            # For red color
            red_mask = cv2.dilate(red_mask, kernel)
            res_red = cv2.bitwise_and(imageFrame, imageFrame,
                                      mask=red_mask)

            # For green color
            green_mask = cv2.dilate(green_mask, kernel)
            res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                        mask=green_mask)

            # For blue color
            blue_mask = cv2.dilate(blue_mask, kernel)
            res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                       mask=blue_mask)

            # For violet color
            violet_mask = cv2.dilate(violet_mask, kernel)
            res_violet = cv2.bitwise_and(imageFrame, imageFrame,
                                         mask=violet_mask)

            # For orange color
            orange_mask = cv2.dilate(orange_mask, kernel)
            res_orange = cv2.bitwise_and(imageFrame, imageFrame,
                                         mask=orange_mask)

            # For brown color
            brown_mask = cv2.dilate(brown_mask, kernel)
            res_brown = cv2.bitwise_and(imageFrame, imageFrame,
                                        mask=brown_mask)

            # Creating contour to track gray color
            contours, hierarchy = cv2.findContours(gray_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (128, 128, 128), 2)#all in RGB

                    cv2.putText(imageFrame, "Gray", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (128, 128, 128))

            # Creating contour to track black color
            contours, hierarchy = cv2.findContours(black_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (0, 0, 0), 2)

                    cv2.putText(imageFrame, "Black", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 0))
            # Creating contour to track white color
            contours, hierarchy = cv2.findContours(white_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (255, 180, 255), 2)

                    cv2.putText(imageFrame, "White", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 180, 255))
            # Creating contour to track yellow color
            contours, hierarchy = cv2.findContours(yellow_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (255, 255, 0), 2)

                    cv2.putText(imageFrame, "Yellow", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 255, 0))

            # Creating contour to track red color
            contours, hierarchy = cv2.findContours(red_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (255, 0, 0), 2)

                    cv2.putText(imageFrame, "Red", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0))

            # Creating contour to track green color
            contours, hierarchy = cv2.findContours(green_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (0, 255, 0), 2)

                    cv2.putText(imageFrame, "Green", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0))

            # Creating contour to track blue color
            contours, hierarchy = cv2.findContours(blue_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (0, 0, 255), 2)

                    cv2.putText(imageFrame, "Blue", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255))

            # Creating contour to track violet color
            contours, hierarchy = cv2.findContours(violet_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (143, 0, 255), 2)

                    cv2.putText(imageFrame, "Violet", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (143, 0, 255))

            # Creating contour to track orange color
            contours, hierarchy = cv2.findContours(orange_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (255, 140, 0), 2)

                    cv2.putText(imageFrame, "Orange", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 140, 0))

            # Creating contour to track brown color
            contours, hierarchy = cv2.findContours(brown_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y),
                                               (x + w, y + h),
                                               (210, 161, 140), 2)

                    cv2.putText(imageFrame, "Brown/Skin", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (210, 161, 140))

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
            x = Image.fromarray(imageFrame)
            s = BytesIO();
            x.save(s, "png")
            result = base64.b64encode(s.getvalue()).decode()
            await self.send(text_data=json.dumps({
                'result': str(result)
            }))