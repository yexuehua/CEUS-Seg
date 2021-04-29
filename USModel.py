import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, multiply, Lambda, add, Activation
from keras.layers import concatenate
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np


# Define loss and performance metrics
# Partially from Abraham and Khan (2019) - A Novel Focal Tversly Loss Function for Lesion Segmentation

# Dice score coefficient and Dice loss
def dsc(y_true, y_pred):
    smooth = 1.
    # masks
    y_true_fm = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_fm * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_fm) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

# Performance metrics: Dice score coefficient, IOU, recall, sensitivity
def auc(y_true, y_pred):
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1)) # ground truth
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    tpr = (tp + K.epsilon()) / (tp + fn + K.epsilon()) #recall
    tnr = (tn + K.epsilon()) / (tn + fp + K.epsilon())
    prec = (tp + K.epsilon()) / (tp + fp + K.epsilon()) #precision
    iou = (tp + K.epsilon()) / (tp + fn + fp + K.epsilon()) #intersection over union
    dsc = (2*tp + K.epsilon()) / (2*tp + fn + fp + K.epsilon()) #dice score
    return [dsc, iou, tpr, prec]

# Expand a tensor by repeating the elements in a dimension
def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

# Convolutional block for UNet
def ConvBlock(in_fmaps, num_fmaps):
    # Inputs: feature maps for UNet, number of output feature maps
    conv1 = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(in_fmaps)
    conv_out = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(conv1)
    return conv_out

# Salient attention block
def SalientAttentionBlock(f_maps, sal_ins, pool_maps, num_fmaps):
    # Inputs: feature maps from UNet, saliency images, pooled layers from UNet, number of output feature maps
    conv1_salins = Conv2D(128, (1, 1), activation='relu')(sal_ins)
    conv1_fmaps = Conv2D(128, (1, 1), strides=(2, 2), activation='relu')(f_maps)
    attn_add = add([conv1_fmaps,conv1_salins])
    conv_1d = Conv2D(128, (3, 3), activation='relu', padding='same')(attn_add)
    conv_1d = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_1d)
    conv_1d = Conv2D(1, (1, 1), activation='relu')(conv_1d)
    conv_1d = expend_as(conv_1d,32)
    conv_nd = Conv2D(num_fmaps, (1, 1), activation='relu')(conv_1d)
    attn_act = Activation('sigmoid')(conv_nd)
    attn = multiply([attn_act, pool_maps])
    return attn

# Convolutional block for UNet
def UNetBlock(in_fmaps, num_fmaps):
    # Inputs: feature maps for UNet, number of output feature maps
    conv1 = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(in_fmaps)
    conv_out = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(conv1)
    return conv_out

# Build the model
def Network(input_size, pretrained_weights=None):

    input = Input(shape=input_size)

    conv1 = ConvBlock(input, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ConvBlock(pool1, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ConvBlock(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ConvBlock(pool3, 64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = ConvBlock(pool4, 128)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = ConvBlock(up6, 64)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = ConvBlock(up7, 64)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = ConvBlock(up8, 32)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = ConvBlock(up9, 32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs = input, outputs = conv10)
    model.compile(optimizer = Adam(lr = 0.0001), loss = dice_loss, metrics = [dsc])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# Build the model
def UNet_SA(input_size, pretrained_weights=None):
    input = Input(shape=input_size)
    conv1 = UNetBlock(input, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    input2 = Input(shape=(input_size[0], input_size[1], 1))

    dwns1 = MaxPooling2D(2,2)(input2)
    attn1 = SalientAttentionBlock(conv1, dwns1, pool1, 32)

    conv2 = UNetBlock(attn1, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    dwns2 = MaxPooling2D(4,4)(input2)
    attn2 = SalientAttentionBlock(conv2, dwns2, pool2, 32)

    conv3 = UNetBlock(attn2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    dwns3 = MaxPooling2D(8,8)(input2)
    attn3 = SalientAttentionBlock(conv3, dwns3, pool3, 64)

    conv4 = UNetBlock(attn3, 64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    dwns4 = MaxPooling2D(16,16)(input2)
    attn4 = SalientAttentionBlock(conv4, dwns4, pool4, 64)

    conv5 = UNetBlock(attn4, 128)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), attn3], axis=3)
    conv6 = UNetBlock(up6, 64)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), attn2], axis=3)
    conv7 = UNetBlock(up7, 64)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), attn1], axis=3)
    conv8 = UNetBlock(up8, 32)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = UNetBlock(up9, 32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs = [input, input2], outputs = conv10)

    model.compile(optimizer = Adam(lr = 0.0001), loss = dice_loss, metrics = [dsc])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
