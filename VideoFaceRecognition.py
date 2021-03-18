import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

## Compute Encodings
## Now that we have a list of images we can iterate through those and create a corresponding encoded list
#  for known faces

def findEncodings(images,num):
    print('Creating Encodings for',num,'reference images detected')
    count = 0
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        count += 1
        print(count,"of",num,"completed")

    return encodeList

## Function for Marking Attendance

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  line:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')
            print("Entry made in Attendance file")

## Now we can simply call this function with the images list as the input arguments.


## Importing Images
## we will write a script to import all images in a given folder
## We will store all the images in one list and their names in another


path = 'TrainImages'
images = []     # LIST CONTAINING ALL THE IMAGES
classNames = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
num = len(myList)


encodeListKnown = findEncodings(images,num)
print('Encodings Complete')

## Capture video from webcam
if cv2.VideoCapture(0):
    print('video capture successfully initiated ...')
cap = cv2.VideoCapture(0)

## The While loop for Webcam Image
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    print('Reading Video: ', success)
    cv2.imshow('Pre-Process video', imgS)

    ## Find encodings for faces in webcam image

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    print('len of facesCurFrame: ',len(facesCurFrame))
    print('len of encodesCurFrame: ', len(encodesCurFrame))
    ## Find Matches to any known face encodings
    print("Working to find matches now ...")

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        print('entered FOR LOOP') ## REMOVE AFTER DEBUG
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        ## Once we have the list of face distances we can find the minimum one, as this would be the best match.
        matchIndex = np.argmin(faceDis)

        ## Now based on the index value we can determine the name of the person and display it on the original Image.

        ## Labeling Unknown faces as well

        if faceDis[matchIndex] < 0.50:
            print('Match Found...')  ## REMOVE AFTER DEBUG
            namenum = classNames[matchIndex].upper()
            name = namenum.split('-')[0]
            print(name)
            markAttendance(name)
        else:
            print('Match NOT Found...')  ## REMOVE AFTER DEBUG
            name = 'Unknown'
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('webcam',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()



