import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('mul.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mul_t.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 6)

#cv2.putText(img_rgb, 'Threshold = ' + str(threshold), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)    
cv2.imwrite('re.jpg',img_rgb)
plt.imshow(img_rgb)
plt.show()