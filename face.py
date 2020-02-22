# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(grey, orig_img):
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(orig_img,(x,y), (x+w,y+h), (255,0,0), 2)
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = orig_img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0), 2)
    
    return orig_img
#video
''' capturing video from camera. '0' for webcam '1' for others '''
video_capture = cv2.VideoCapture(0)
while True:
    _, orig_img = video_capture.read()
    grey = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    canvas = detect(grey, orig_img)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
