
import cv2 as cv
import numpy as np

print('Building a Real time Face detection using open cv')


# loading haar cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
#smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

cap = cv.VideoCapture(0)

while True:

    ret,frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # use scaleFactor = 1.05 and minNeighbors = 5for more accuracy
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    print(f'number of Faces Detected = {len(faces)}')

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),thickness=2)
        cv.putText(frame, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    cv.imshow('Detected Faces',frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()




