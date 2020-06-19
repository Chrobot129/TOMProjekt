#%%
from starter_code.utils import load_segmentation
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import numexpr as ne
from joblib import Parallel, delayed
import multiprocessing
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
import pandas as pd
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from skimage import color
import cv2
from IPython.display import clear_output
#from tensorflow.examples.models.pix2pix import pix2pix


#%%

def get_data(case_nr):

    def get_case(case_nr):
        
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\preprocessed_added_dummy")

        case_nb = nibabel.load('case{}_preprocessed_added.nii.gz'.format(case_nr))
        case = case_nb.get_fdata()

        return case

    def get_segment(case_nr):
        
        os.chdir("c:\\Users\\Chrobot\\Desktop\\TOM\\Projekt\\kits19\\segment_added_dummy")

        segment_nb = nibabel.load('segmentation{}_added.nii.gz'.format(case_nr))
        segment = segment_nb.get_fdata()

        return segment

    X = get_case(case_nr)
    Y = get_segment(case_nr)
    return X,Y

#%%
base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
OUTPUT_CHANNELS = 3
#%%
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[512, 512, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# %%
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)
#%%
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

#%%
image, mask = get_data(4)
image = image[20,:,:]
mask = mask[20,:,:]

sample_image = color.gray2rgb(image)
sample_mask = color.gray2rgb(mask)

#%%

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

#%%
train_data, train_mask  = get_data(4)
test_data, test_mask = get_data(5)

train_data = np.stack((train_data,train_data,train_data), axis = 3).astype(np.float16)
train_mask = np.stack((train_mask,train_mask,train_mask), axis = 3).astype(np.float16)
test_data = np.stack((test_data,test_data,test_data), axis = 3).astype(np.float16)
test_mask = np.stack((test_mask,test_mask,test_mask), axis = 3).astype(np.float16)

train_dataset = (train_data,train_mask) 
test_dataset = (test_data, test_mask)
#%%
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


BATCH_SIZE = 139
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 834// BATCH_SIZE

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 834//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

# %%
