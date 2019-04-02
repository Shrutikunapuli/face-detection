import cv2
import sys
import sqlite3

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#path if neccesary 

video_capture = cv2.VideoCapture(0)
Id=input('enter user ID')


sampleNum=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        sampleNum=sampleNum+1
        # give your path here 
        cv2.imwrite('#give your path here'+str(sampleNum)+"."+str(Id)+".jpg",gray[y:y+h,x:x+w])
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if (sampleNum)>30:
            break
 
    

print(Id)
print(sampleNum)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
