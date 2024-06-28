import cv2;
import os


#--------------------------------1 Generating dataset--------------------------------
def generate_dataset(id):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #convert the image to gray scale
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        #scaling factor = 1.3
        #Minimum Neighbors = 5

        #detect face and crop the face
        if faces is ():
            #if face not detected return None
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    #initialize webcam
    cap = cv2.VideoCapture(0)   
    
    #initialize the count of the sample
    img_id=0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #save file in specified directory with unique name
            file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)

            #put count on images and display live count
            #(50,50) is the position of the text
            #(0,255,0) is the color of the text
            #(1) is the font of the text
            #(2) is the thickness of the text
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Cropped face", face)

        if cv2.waitKey(1)==13 or int(img_id)==200:
            break

    #release the camera
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")


# calling the generate_dataset function
generate_dataset(2)
