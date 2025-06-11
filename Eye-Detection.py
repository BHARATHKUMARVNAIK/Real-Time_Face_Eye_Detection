
import cv2 as cv
import numpy as np
import time

print('Building Eye detection using open-cv')


# Loading haar-cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
#smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

cap = cv.VideoCapture(0)
save_counts = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f'Number of faces is : {len(faces)}')


    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv.putText(frame, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=10)
        print(f'Number of eyes for face : {len(eyes)}')

        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,255), 2)
            cv.putText(roi_color, 'Eye', (ex, ey - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


        # save image
        if len(eyes) >= 1 and save_counts == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename =  f'face eye detected_{timestamp}.png'
            cv.imwrite(filename, frame)
            print(f'Image saved as : {filename}')
            save_counts += 1


    cv.imshow('Detected Face and Eye', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



