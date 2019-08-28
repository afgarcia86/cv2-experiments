import cv2
import numpy as np


for i in ['0', '1', '2', '3', '4']:
	img_rgb = cv2.imread('images/screenshot.jpg')
	template = cv2.imread(f'images/{i}.jpg')
	w, h = template.shape[:-1]

	print(f'Testing {i}')

	try:
		result = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

		#the get the best match fast use this:
		(min_x, max_y, minloc, maxloc) = cv2.minMaxLoc(result)
		# (x,y) = minloc

		cv2.rectangle(img_rgb, maxloc, (maxloc[0] + h, maxloc[1] + w), (0, 255, 0), 2)

		# threshold = 0
		# loc = np.where(res >= threshold)
		# first = True
		# for pt in zip(*loc[::-1]):  # Switch collumns and rows
		# 	if first:
		# 		cv2.rectangle(img_rgb, pt, (pt[0] + h, pt[1] + w), (0, 255, 0), 2)
		# 	first = False

		cv2.imshow('template', template)
		cv2.imshow('result', img_rgb)
		cv2.waitKey()
		cv2.destroyAllWindows()
	except Exception as e:
		print(f'Failed {i} - {e}')
		continue
