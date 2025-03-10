import os

import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from keras.api.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

import matplotlib as mpl 

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models.h5')
model = load_model(model_path)
height, width = 299, 299
class_names = {0: 'Bacterial Blight', 1: 'Blast', 2: 'Brown Spot', 3: 'Tungro'}

def img_to_array(img_path):
  img = image.load_img(img_path, target_size=(height, width))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)
  return img_array
  
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap, preds
  
def get_heatmap(img_path, this_model = model, pred_id = None):
  last_conv_layer_name = "mixed10"

  this_model.layers[-1].activation = None
  
  img_arr = img_to_array(img_path)
  heatmap, preds = make_gradcam_heatmap(img_arr, this_model, last_conv_layer_name, pred_id)

  pred_idx = np.argmax(preds[0])
  confidence_level = tf.reduce_max(tf.nn.softmax(preds[0])).numpy()
  score = round(confidence_level * 100, 3)
  
  label = [class_names[pred_idx], str(score)]
    
  return heatmap, label

def save_and_display_gradcam(img_path, heatmap, max_width=None, cam_path="result.jpg", alpha=0.5,):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    
    return cam_path