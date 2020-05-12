import argparse
import numpy as np
import time

from utils.config import Config
from model import ICNet, ICNet_BN
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

tf.disable_eager_execution()

model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'ade20k'
filter_scale = 2

class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)

    # You can choose different model here, see "model_config" dictionary. If you choose "others",
    # you need to set "filter_scale" to 2, otherwise set it to 1
    model_type = 'others'

    # Set pre-trained weights here (You can download weight from Google Drive)
    model_weight = './checkpoint/model.ckpt-27150'

    # Define default input size here
    INFER_SIZE = (256, 256, 3)

cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()

# Create graph here
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)

im1 = np.random.uniform(0, 256, [cfg.INFER_SIZE[1], cfg.INFER_SIZE[0], 3])

results = net.predict(im1)

builder = tf.saved_model.builder.SavedModelBuilder('./icnet_' + dataset + '_saved')

sigs = {}

g = tf.get_default_graph()
inp = g.get_tensor_by_name("Placeholder:0")
out = g.get_tensor_by_name("crop_to_bounding_box/Slice:0")

sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    tf.saved_model.signature_def_utils.predict_signature_def(
        {"in": inp}, {"out": out})

builder.add_meta_graph_and_variables(net.sess,
                                     [tag_constants.SERVING],
                                     signature_def_map=sigs)

builder.save()

