#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array import array

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')
parser.add_argument('--mode', dest='mode', type=str, default='live', help='[live or file]')

args = parser.parse_args()

fx_320_240 = 375.0
fy_320_240 = 375.0
cx_320_240 = 160.0
cy_320_240 = 120.0
bf_320_240 = 375.0

fx_960_720 = 1125.0
fy_960_720 = 1125.0
cx_960_720 = 480.0
cy_960_720 = 360.0
bf_960_720 = 1125.0

def main(_):
  print('In main')
  with tf.Graph().as_default():
    height = args.height
    width = args.width
    mode = args.mode
    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

    with tf.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    print('created model saver')
    if mode == 'live':
      cam = cv2.VideoCapture(0)
      print('captured video!')
    elif mode == 'file':
      path = 'test_data'
      files = []
      # r=root, d=directories, f = files
      for r, d, f in os.walk(path):
          for file in f:
              if '.png' in file:
                  files.append(os.path.join(r, file))
      files = sorted(files)
      print('{0} images to process'.format(len(files)))

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        image_sequence = 0
        while True:
          if mode == 'live':
            for i in range(4):
              cam.grab()
            ret_val, img = cam.read()
          elif mode == 'file':
            if image_sequence >= len(files):
              break
            img = cv2.imread(files[image_sequence])
            image_sequence = image_sequence + 1
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          start = time.time()
          disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()

          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (int(width/2), height))

          cv2.imshow('pydnet', toShow)
          k = cv2.waitKey(1)
          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          cloud = point_cloud(disp, cx_960_720, cy_960_720, fx_960_720, fy_960_720, bf_960_720)
          ax.scatter(cloud[0], cloud[1], cloud[2], s=1)
          ax.set_xlabel('X Label')
          ax.set_ylabel('Y Label')
          ax.set_zlabel('Z Label')
          plt.show()
#          if mode == 'file':
#            f = open('PointCloud/' + str(image_sequence) + '.bin', 'w+b')
#            byte_arr = []
#            for i in range(len(cloud[0])):
#              byte_arr.append(float(cloud[0][i]))
#              byte_arr.append(float(cloud[1][i]))
#              byte_arr.append(float(cloud[2][i]))
#            float_array = array('d', byte_arr)
#            float_array.tofile(f)
#            f.close()
#            image_sequence = image_sequence + 1

          if k == 1048603 or k == 27:
            break  # esc to quit
          if k == 1048688:
            cv2.waitKey(0) # 'p' to pause

          print("Time: " + str(end - start))
          del img
          del disp
          del toShow
        if mode == 'live':
          cam.release()

def point_cloud(depth, cx, cy, fx, fy, bf):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    print(depth.shape)
    depth = depth[0, :, :, 0]
    print(depth.shape)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    max_val = np.amax(depth)
    min_val = np.amin(depth)
    print('max: {0}, min: {1}'.format(max_val, min_val))
    #depth = (depth - min_val) / (max_val - min_val) * 255
    depth = depth * bf
    valid = (depth > 0)

    result_x = []
    result_y = []
    result_z = []
    for i in range(rows):
      for j in range(cols):
        z = depth[i][j]
        if z <= 0:
          result_x.append(0.0)
          result_y.append(0.0)
          result_z.append(0.0)
        else:
          x = z * (i - cx) / fx
          y = z * (j - cy) / fy
          result_x.append(x)
          result_y.append(y)
          result_z.append(z)
#    z = np.where(valid, depth, np.nan)
#    x = np.where(valid, z * (c - cx) / fx, 0)
#    y = np.where(valid, z * (r - cy) / fy, 0)
    return (result_x, result_y, result_z)

if __name__ == '__main__':
    tf.app.run()
