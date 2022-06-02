import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
from yolov5.utils.general import increment_path

def face(image_path,n):  
  detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
  img_path = image_path
  
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  dets = detector(img, upsample_num_times=1)


  for i, d in enumerate(dets):
      x1, y1 = d.rect.left(), d.rect.top()
      if x1<0 : x1=0
      if y1<0 : y1=0
      x2, y2 = d.rect.right(), d.rect.bottom()

      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      roi=img[y1:y2,x1:x2]
      cv2.imwrite(str(increment_path('roi/{}/roi.jpg'.format(n)).with_suffix('.jpg')),roi)
      break
    