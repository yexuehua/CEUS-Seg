from USPreprocessing import *
from USModel import *
import os
import cv2
import pandas as pd

# Introduce parameters
img_row = 768
img_col = 512
img_chan = 3
epochnum = 200
batchnum = 4
input_size = (img_row, img_col, img_chan)

def create_saliant_map():
    path_imgs = "./dataset/test/image"

    imagesList = os.listdir(path_imgs)
    CEUS_images, US_images = img_load(path_imgs, imagesList)
    model = Network(input_size, "./ModelCheckpoint/unet-withoutaug/KFold3/checkpoint/wights.198.hdf5")
    preds = model.predict(CEUS_images[:len(imagesList)//2])
    preds_part2 = model.predict(CEUS_images[len(imagesList)//2:])

    preds = preds*255
    preds_part2 = preds_part2*255

    for i in range(len(preds)):
        display_img_mask(CEUS_images, CEUS_images, preds, i)

    for idx, pred_img in enumerate(preds):
        cv2.imwrite("./Dataset/test/salient/" + imagesList[idx]+".jpg", pred_img[:,:,0])
    idx2 = idx
    for pred_img in preds_part2:
        idx2 += 1
        cv2.imwrite("./Dataset/test/salient/" + imagesList[idx2]+".jpg", pred_img[:,:,0])
create_saliant_map()
"""
# Directory with images and ground truths
path_imgs = "./dataset/image"
path_masks = "./dataset/mask"

imagesList = os.listdir(path_imgs)
masksList = [i+"_Merge.nii" for i in imagesList]

CEUS_images, US_images = img_load(path_imgs, imagesList)
CEUS_masks, US_masks = img_load(path_masks, masksList)

df_test = pd.read_csv("df_test_image.csv")
#test_name = df_test["0"][2].strip("[").strip("]").split('\\').strip("""""")
#print(test_name)
test_index = list(map(int, df_test["index"][2].strip("[").strip("]").strip().split(" ")))
test_index = np.array(test_index)

test_images, test_masks = CEUS_images[test_index], CEUS_masks[test_index]

model = Network(input_size, "./ModelCheckpoint/KFold2/checkpoint/wights.198.hdf5")
preds = model.predict(test_images)

for i in range(len(test_masks)):
    display_img_mask(test_images, test_masks, preds, i)
"""

