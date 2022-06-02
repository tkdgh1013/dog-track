from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import pickle
import numpy as np
import skimage.io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import tensorflow.keras.backend as K
#from triplets_processing import *
def cluster():
  PATH='roi/'
  """
  PATH = '../data/dogfacenet/aligned/after_1/'
  PATH_SAVE = '../output/history/'
  PATH_MODEL = '../output/model/'
  """
  SIZE = (224,224,3)
  VALID_SPLIT = 0.1
  TEST_SPLIT = 1

  filenames = np.empty(0)
  labels = np.empty(0)
  idx = 0
  for root,dirs,files in os.walk(PATH):
      if len(files)>0:
          for i in range(len(files)):
              files[i] = root + '/' + files[i]
          filenames = np.append(filenames,files)
          #labels = np.append(labels,np.ones(len(files))*idx)
          labels=np.append(labels,np.int64(os.listdir(PATH)[idx]))
          idx += 1
  
  
  h,w,c = SIZE
  images = np.empty((len(filenames),h,w,c))
  for i,f in enumerate(filenames):
      images[i] = resize(skimage.io.imread(f),SIZE)
  
  # Normalization
  #images /= 255.0

  nbof_classes = len(np.unique(labels))

  nbof_test = int(TEST_SPLIT*nbof_classes)


  keep_test = np.less(labels,np.int64(np.max(labels))+1)

  images_test = images[keep_test]
  labels_test = labels[keep_test]
  
  #images_train = images[keep_train]
  #labels_train = labels[keep_train]

  alpha = 0.3
  def triplet(y_true,y_pred):
      
      a = y_pred[0::3]
      p = y_pred[1::3]
      n = y_pred[2::3]
      
      ap = K.sum(K.square(a-p),-1)
      an = K.sum(K.square(a-n),-1)

      return K.sum(tf.nn.relu(ap - an + alpha))

  def triplet_acc(y_true,y_pred):
      a = y_pred[0::3]
      p = y_pred[1::3]
      n = y_pred[2::3]
      
      ap = K.sum(K.square(a-p),-1)
      an = K.sum(K.square(a-n),-1)
      
      return K.less(ap+alpha,an)

  
  model = tf.keras.models.load_model('2019.07.29.dogfacenet.249.h5', custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})


  from sklearn.preprocessing import MultiLabelBinarizer





  mod = tf.keras.Model(model.layers[0].input, model.layers[-1].output)
  predict=mod.predict(images_test)
  y_prob = mod.predict(images_test, verbose=0) 
  predicted2 = y_prob.argmax(axis=-1)


  from sklearn.cluster import KMeans

    #silhouette
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_samples


  silhouette_vals=[]
  for i in range(2,len(np.unique(predicted2))):
    kmeans_plus=KMeans(n_clusters=i,init='k-means++')
    pred=kmeans_plus.fit_predict(predict)
    silhouette_vals.append(np.mean(silhouette_samples(predict,pred,metric='euclidean')))
  
  if len(silhouette_vals)==0:
    return [[labels_test[i] for i in range(len(labels_test))]]


  kmeans = KMeans(n_clusters=silhouette_vals.index(max(silhouette_vals))+2,max_iter=2000, random_state=0,tol=0.576).fit(predict)
  images_cluster = [images_test[np.equal(kmeans.labels_,i)] for i in range(len(labels_test))]
  labels_cluster = [labels_test[np.equal(kmeans.labels_,i)] for i in range(len(labels_test))]


  
  for i in range(len(images_cluster)):
      length = len(images_cluster[i])
      
      if length > 0:
          """
          print(labels_cluster[i])
          fig=plt.figure(figsize=(length*2,2))
          for j in range(length):
              plt.subplot(1,length,j+1)
              plt.imshow(images_cluster[i][j])
              plt.xticks([])
              plt.yticks([])
          plt.show()
          """
      else:
        labels_cluster.pop()

  return labels_cluster