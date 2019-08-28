import numpy as np
import operator
import cv2
from matplotlib import pyplot as plt


regions = {
    '1': {'x1': 0, 'y1': 410, 'x2': 354, 'y2': 508},
    '2': {'x1': 0, 'y1': 0, 'x2': 354, 'y2': 255},
    '3': {'x1': 120, 'y1': 255, 'x2': 238, 'y2': 410},
    '4': {'x1': 0, 'y1': 255, 'x2': 120, 'y2': 410},
    '0': {'x1': 238, 'y1': 255, 'x2': 354, 'y2': 410},
}


img2 = cv2.imread('images/page.jpg')  # target Image
# img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


for i in ['0', '1', '2', '3', '4']:
    img1 = cv2.imread(f'images/ugc/{i}.jpg')          # query Image
    # img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if i == 'test':
        i = '0'

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB

    h, w = img1.shape[:2]
    y, x = int(h/2), int(w/2)
    # roi = img1[y-200:600, x-200:400]

    roi = 200
    mask = None
    if roi:
        # create a mask image filled with zeros, the size of original image
        mask = np.zeros(img1.shape[:2], dtype=np.uint8)
        # draw your selected ROI on the mask image
        cv2.rectangle(mask, (x-roi,y-roi), (x+roi,y+roi), (255), thickness = -1)

    # cv2.imshow("cropped", mask)
    # cv2.waitKey(0)
    # continue

    kp1, des1 = orb.detectAndCompute(img1,mask)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    good_matches = matches[:20]

    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    x = { d: np.mean([(c['x1'] < p[0][0] < c['x2']) and (c['y1'] < p[0][1] < c['y2']) for p in dst_pts]) for d, c in regions.items()}
    v =  max(x.items(), key=operator.itemgetter(1))[0]
    answer = regions[v]

    coords = [
                [[w + answer['x1'], answer['y1']]],
                [[w + answer['x2'], answer['y1']]],
                [[w + answer['x2'], answer['y2']]],
                [[w + answer['x1'], answer['y2']]]
            ]

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None, flags = 2)

    # Draw bounding box in Red
    img3 = cv2.polylines(img3, [np.int32(coords)], True, (0,255,0) if v == i else (0,0,255), 5, cv2.LINE_AA)

    cv2.imshow("result", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # or another option for display output
    # plt.imshow(img3, 'result'), plt.show()