# 取色器
# http://www.jiniannet.com/Page/allcolor
# 将获取的四个点的rgb值填入下方的rgb，即可得到四个点的hsv上下限
import cv2
import numpy as np

rgb = '#1A201C,#1E1A12,#252E2A,#1B231A'

rgb = rgb.split(',')

# 转换为BGR格式，并将16进制转换为10进制
bgr = [[int(r[5:7], 16), int(r[3:5], 16), int(r[1:3], 16)] for r in rgb]

# 转换为HSV格式
hsv = [list(cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2HSV)[0][0]) for b in bgr]

hsv = np.array(hsv)
print('H:', min(hsv[:, 0]), max(hsv[:, 0]))
print('S:', min(hsv[:, 1]), max(hsv[:, 1]))
print('V:', min(hsv[:, 2]), max(hsv[:, 2]))
