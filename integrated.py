import cv2
import numpy as np
import smtplib
import playsound
import threading
import imutils
import datetime
from playsound import playsound







Alarm_Status = False
Email_Status = False
Fire_Reported = 0

gun_cascade = cv2.CascadeClassifier('cascade.xml')

camera = cv2.VideoCapture(0)

firstFrame = None
gun_exist = False


while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = cv2.resize(frame, (960, 540))
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)
    if int(no_red) > 15000:
        Fire_Reported = Fire_Reported + 1
    cv2.imshow("FIRE", output)
    if Fire_Reported >= 1:
        if Alarm_Status == False:
            print("Warning There's a fire")
            playsound('C:/Users/polit/OneDrive/Desktop/hackaccino/alarm-sound.mp3')
            Alarm_Status=True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))
    if len(gun) > 0:
        gun_exist = True

    for (x,y,w,h) in gun:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    if firstFrame is None:
        firstFrame = gray
        continue
        
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("ARMS", frame)
    key = cv2.waitKey(1) & 0xFF

if gun_exist:
    print("WARNING!!! guns detected")

else:
    print("guns NOT detected")

camera.release()
cv2.destroyAllWindows()