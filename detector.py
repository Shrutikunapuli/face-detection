import cv2
import numpy as np

recognizer = cv2.cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
#test = ('haarcascade_frontalface_default.xml')
cascadePath = ("haarcascade_frontalface_default.xml")



faceCascade = cv2.CascadeClassifier(cascadePath)


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_TRIPLEX
fontcolor = (255, 0, 0)
while True:
    ret, im = cam.read()
    if ret is True:

       gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        continue
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if (conf<5):
        if (Id == 1):
                Id = "shruti"
        elif (Id == 2):
                Id = "venkat"
        else:
                Id = "Unknown"
        print(Id)
        cv2.putText(im, str(Id), (x, y + h), font, 1,fontcolor)
    cv2.imshow('im', im)
    if cv2.waitKey(10) &  0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
