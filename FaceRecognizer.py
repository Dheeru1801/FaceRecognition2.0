import cv2;
import numpy as np
from PIL import Image
import os


#--------------------------------3 Recognize the face--------------------------------

#3a Function to draw boundary around the detected face
#crop face->convert to gray scale ->give the cropped image to the classifier
#To draw rectangle , get the real image from the webcam
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    #convert the image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect the face in the image
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    coords = []

    for (x, y, w, h) in features:

        #draw rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        #predict the id of the user
        id, pred = clf.predict(gray_img[y:y+h, x:x+w])

        #calculate the confidence
        confidence=int(100*(1-pred/300))
        
        #check the confidence of the prediction
        if confidence>70:
            if id==1:
                cv2.putText(img,"Dheeraj", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id==2:
                cv2.putText(img,"Mahesh", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

        coords = [x, y, w, h]

    return coords
    
#3b Function to recognize the person in the rectangle around the face
def recognize(img, clf, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["blue"], "Face", clf)
    return img


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")


#initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face detection", img)

    if cv2.waitKey(1)==13:
        break
    if cv2.getWindowProperty("Face detection", cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()