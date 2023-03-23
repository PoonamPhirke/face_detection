
import cv2
import os
import pickle
import numpy as np
import cvzone

import face_recognition

#1. CODE TO RUN WEBCAM
cap = cv2.VideoCapture(0) # to capture video.
cap.set(3, 640)# width of webcam screen. Graphics is based on these dimension.
cap.set(4,480) # height of webcam screen.
#2.  GRAPHICS
imgBackground = cv2.imread('resources/background.png')

# Import modes . As similarly to the array input we will take input of images (mode) and store it into python list
#Importing the mode images into a list
folderModePath = 'resources/Modes' # we are giving this path to the list to call image(mode)
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

#print(len(imgModeList)) # tells whether the images are imported or not

# Face recognition
# Load the encoding files
print("Loading the encoded file")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file) # it will add all the list and info into this encode list with ids
file.close
encodeListKnown, studentIds = encodeListKnownWithIds
#print(studentIds)
print("Encode file loaded....")

while True:
    success, img = cap.read() # to read

    # face recognition step 3
    # Making images smaller..scale it down to 1/4 th of the size
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# BGR to RGB

    faceCurFrame = face_recognition.face_locations(imgS)
    #find encodins of new faces not all image
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)




    # to overlap camera interface on background
    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0] # copy modes into background # gives us image.. We are dynamic values here ..we can add variable ..which show us stages

    #compare
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        # find matches
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)
        # Extract these values
        matchIndex = np.argmin(faceDis)
        print("Match Index ", matchIndex)

       # if matches[matchIndex]:
           # print("Known Face detected")
           # print(studentIds[matchIndex])
            # show rectangle around face provided by opencv but we are using cvzone
            #y1, x2, y2, x1 = faceLoc   #create bounding box
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            #bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            #imgBackground = cvzone.cornerRect(imgBackground,bbox, rt=0)# bounding box




    #cv2.imshow("Webcam", img)
    cv2.imshow("Voting System ",imgBackground) # Actual output
    cv2.waitKey(1)




