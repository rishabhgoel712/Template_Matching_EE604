
import numpy as np
import cv2
from matplotlib import pyplot as plt

c_img = cv2.imread('s.jpg',1)
img1 = cv2.imread('s.jpg',0)          # queryImage
img2 = cv2.imread('t.jpg',0) # trainImage
height, width = img2.shape[:2]
# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create
sift = cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
ptx=[]
pty=[]

r = 0.6 #(Ratio)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < r*n.distance:
        good.append([m])
        z=m.queryIdx
        ptx.append(kp1[z].pt[0])
        pty.append(kp1[z].pt[1])
        
ptx.sort()
pty.sort()
np=len(ptx)
midx=ptx[int(np/2)]
midy=pty[int(np/2)]
cv2.rectangle(c_img,(int(midx-width/2),int(midy-height/2)),(int(midx+width/2),int(midy+height/2)) , (255, 255, 255), 10)
print(ptx)
print(pty)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(c_img,kp1,img2,kp2,good,None,flags=2)
#cv2.putText(img3, 'Ratio = ' + str(r), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)

plt.imshow(img3),plt.show()
cv2.imwrite(str(r) + 'Result.jpg',img3)