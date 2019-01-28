# Read a Video Stream from Camera (Frame by Frame)
import cv2

cap=cv2.VideoCapture(0)
# 0 stands for default camera
face_cascade=cv2.CascadeClassifier('../Xml Files/haarcascade_frontalface_alt.xml')

while True:
	ret,frame=cap.read()
	# ret is boolean and symbol of frame captured properly or not
	print(ret)
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)
	# 1.3 is Scaling factor
	# 5 is Number of neighbors

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		# it shows a rectangle detecting the face

	cv2.imshow('Video Frame',frame)
	#cv2.imshow('Gray Frame',gray_frame)

	#Wait for user input q then you will stop the loop
	key_pressed=cv2.waitKey(1)&0xFF
	# 1 means wait for 1 ms
	# cv2.waitKey will return 32 bit value
	# and operation will 1111 1111
	# ord() will give ascii value of q
	if key_pressed==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
