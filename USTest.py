from USPreprocessing import *
from USModel import *
import os
import pandas as pd

# Introduce parameters
img_row = 768
img_col = 512
img_chan = 3
epochnum = 200
batchnum = 4
input_size = (img_row, img_col, img_chan)

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


