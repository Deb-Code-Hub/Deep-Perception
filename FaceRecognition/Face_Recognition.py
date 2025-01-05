import face_recognition
import numpy as np
import cv2

window_name = "Window"
interframe_wait_ms=10

# Capturing video through webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
color1=list(np.random.choice(range(255),size=3))
color=list(np.random.choice(range(255),size=3))



#colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load first sample picture and learn how to recognize it.
Debraj_image = face_recognition.load_image_file("Debraj.jpg")
Debraj_face_encoding = face_recognition.face_encodings(Debraj_image)[0]

# Load a second sample picture and learn how to recognize it.
Damodar_image = face_recognition.load_image_file("Damodar.jpg")
Damodar_face_encoding = face_recognition.face_encodings(Damodar_image)[0]

# Load a third sample picture and learn how to recognize it.
Renish_image = face_recognition.load_image_file("Renish.jpg")
Renish_face_encoding = face_recognition.face_encodings(Renish_image)[0]

# Load a fourth sample picture and learn how to recognize it.
Chiranjit_image = face_recognition.load_image_file("Chiranjit.jpg")
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

while True:
    _,frame=webcam.read()
    #print(frame)

    rgb_frame=frame[:, :, ::-1]
    #print(rgb_frame)

    face_locations = face_recognition.face_locations(rgb_frame)#rgb_frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Cannot Recognize"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]


        cv2.rectangle(frame, (left, top - 35), (right, bottom), (int(color[0]), int(color[1]), int(color[2])), 2)

        cv2.putText(frame, name, (left+6, top-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (int(color1[0]), int(color1[1]), int(color1[2])), 1)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(interframe_wait_ms) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break


