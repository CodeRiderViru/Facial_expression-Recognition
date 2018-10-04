import numpy as np
import cv2
import os
import time

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def crop_face(clahe_image, face):
	for (x, y, w, h) in face:
		faceslice = clahe_image[y:y+h, x:x+w]
		faceslice = cv2.resize(faceslice, (350, 350))
	return faceslice
	
emotions={"anger" ,"happy","sad","nuetral"}
for emotion in emotions:
	i=1
	print("Give expression of %s emotion" %(emotion))
	while True:
		ret,frame=cap.read()
		#cv2.imshow("preview",frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		clahe_image = clahe.apply(gray) 
		face = face_cascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
		for (x, y, w, h) in face:
			img=cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
				
		cv2.imshow("img",img)
		key=cv2.waitKey(40)
		if key==27:
			break
		if key==32:
			faceslice = crop_face(clahe_image, face)
			#cv2.imshow(str(i),faceslice)
			cv2.imwrite(os.path.join('dataset/%s/' %(emotion) , str(i)+'.jpg'), faceslice)
			print(i)
			i=i+1
	
		
cap.release()
cv2.destroyAllWindows()
