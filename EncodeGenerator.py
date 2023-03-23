# To generate all the encodings that we need of the faces
import cv2
import face_recognition
import pickle #To dump images
import os


#To import all the images of the faces

#Importing the student images
folderPath = 'images' # we are giving this path to the list to call image(mode)
pathList = os.listdir(folderPath) # Get a list of the directory
print(pathList)
imgList = [] # Get the list of all the images
# to extract ids
studentIds = [] # To import students IDs
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    # To get only ID part and to remove the png

    studentIds.append(os.path.splitext(path)[0])
    print(path)
    print(os.path.splitext(path)[0])  # to remove png because we want only id
print(studentIds)  # to check student ids to import it correctly


# to generate encodings and split out all list with encodings
def findEncodings(imageList): # to find encodings
    encodeList = [] # Create list
    for img in imageList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # STEP 1 ... BGR TO RGB
        encode = face_recognition.face_encodings(img)[0] # step 2 Find encodings
        encodeList.append(encode) # It will loop through all yhe images and it will save

    return encodeList # we will return encodelist ..theat will generate all our encodings
print("Encodings started")
# LETS call this function
encodeListKnown = findEncodings(imgList) # generate encodings
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(encodeListKnown)
print("Encodings completed")

# To generate pickle file
file = open("encodeFile.p",'wb') # send in or dump the list in this file
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("file saved")

