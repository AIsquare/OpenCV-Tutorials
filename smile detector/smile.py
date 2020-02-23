# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:51:24 2020

@author: pc
"""
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(grey,orig_img):
    faces = face_cascade.detectMultiScale(grey,1.1,5)
    for (x,y,w,h) in faces:
        cv2.retangle(orig_img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = orig_img[y:y+h,x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_grey, 1.7, 22)

        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0))
        
        eye = eye_cascade.detectMultiscale(roi_grey,1.1,22)
    
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    return orig_img

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
        