import cv2

img=cv2.imread('../Pictures/dog.png')
cv2.imshow('Dog Image',img)
gray=cv2.imread('../Pictures/dog.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray Dog Image',gray)
# by default RGB cv2.imshow()
cv2.waitKey(0)
# 0 means wait for infinite time it can be 200 ms also 
cv2.destroyAllWindows()
