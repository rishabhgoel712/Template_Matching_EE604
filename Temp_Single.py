import cv2
import numpy as np
from matplotlib import pyplot as plt

colored_img = cv2.imread('lib.jpg',1)
colored_tmp = cv2.imread('lib_t.jpg',1)
img = cv2.imread('lib.jpg',0)
img2 = img.copy()
template = cv2.imread('lib_t.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#methods = ['cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    colored_img_ = colored_img.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(colored_img_,top_left, bottom_right, (255,255,255), 10)
    #cv2.putText(colored_img_, meth, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.imwrite(meth + '.jpg',colored_img_)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(cv2.cvtColor(colored_img_, cv2.COLOR_BGR2RGB))
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()