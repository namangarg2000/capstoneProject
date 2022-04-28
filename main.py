import cv2
import numpy as np
import face_recognition


imgMe = face_recognition.load_image_file('Images/me.jpeg')#loading image
imgMe = cv2.cvtColor(imgMe, cv2.COLOR_BGR2RGB)#converting it to RGB
imgTest = face_recognition.load_image_file('Images/me2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMe)[0]#detect face location
encodeElon = face_recognition.face_encodings(imgMe)[0]#encoding image (top,right,bottom,left)
cv2.rectangle(imgMe, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)#color,thickness creating square

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest) #comaparing faces and testing distances
faceDis = face_recognition.face_distance([encodeElon], encodeTest)#find simillarity
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Muskan', imgMe)
cv2.imshow('Muskan', imgTest)
cv2.waitKey(0)






