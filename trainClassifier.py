import os
import numpy as np
from PIL import Image
import cv2


#------------------------------2Training the classifier and save it--------------------------
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        #convert the image to gray scale
        img = Image.open(image).convert('L')
        #convert the image to numpy array format
        imageNp = np.array(img, 'uint8')
        #get the user id of the image by splitting the image name
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    #Train the classifier using LBPHFacerecognizer and save it
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Training completed successfully...")


#calling the train_classifier function
train_classifier("data")