#!/usr/bin/python3

import cv2
import numpy as np
import tensorflow as tf
loaded = tf.saved_model.load('./icnet_ade20k_saved')
infer = loaded.signatures['serving_default']

img = cv2.imread('test.jpg')
h = img.shape[0]
w = img.shape[1]

img = cv2.resize(img, (256, 256))

img_tensor = tf.dtypes.cast(tf.constant(img), tf.float32)

pred = infer(img_tensor)['out'].numpy()
pred = np.squeeze(pred)

print(pred.shape)
pred = cv2.resize(pred, (w, h))
classes = np.argmax(pred, axis=2)
classes = np.uint8(np.squeeze(classes))
classes = cv2.cvtColor(classes, cv2.COLOR_GRAY2RGB)

lut = cv2.imread('ade20k.png')

classes = cv2.LUT(classes, lut)

cv2.imshow('classes', classes)
cv2.waitKey(0)
