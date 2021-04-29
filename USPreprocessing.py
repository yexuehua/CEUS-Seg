import os
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from keras.preprocessing.image import ImageDataGenerator

# Load the all images without any preprocessing
def img_load(dir_path, imgs_list):
    #inital the list
    CEUS_data_list = []
    US_data_list = []
    for i in range(len(imgs_list)):
        # read dicom data
        itkimg_f1 = sitk.ReadImage(os.path.join(dir_path, imgs_list[i]))
        f1_data = sitk.GetArrayFromImage(itkimg_f1)
        if np.max(f1_data)>1:
            f1_data = f1_data/255
        if len(f1_data.shape)==4:
            # get the shape data
            _, h, w, c = f1_data.shape
            # inital the data
            CEUSImage = np.zeros((h, w // 2, c))
            USImage = np.zeros_like(CEUSImage)
            # extract content
            CEUSImage = f1_data[0][:, 0:w // 2, :]
            USImage = f1_data[0][:, w // 2:, :]
        elif len(f1_data.shape)==3:
            # get the shape data
            _, h, w = f1_data.shape
            # inital the data
            CEUSImage = np.zeros((h, w // 2))
            USImage = np.zeros_like(CEUSImage)
            # extract content
            half_w = w//2
            CEUSImage = f1_data[0][:, 0:half_w]
            USImage = f1_data[0][:, half_w:]
            CEUSImage = np.expand_dims(CEUSImage, axis=2)
            USImage = np.expand_dims(USImage, axis=2)
        # store the data to a list
        CEUS_data_list.append(CEUSImage)
        US_data_list.append(USImage)

    CEUS_data = np.array(list(CEUS_data_list))
    US_data = np.array(list(US_data_list))
    # Expand the dimensions of the arrays
    # imgs_array = np.expand_dims(imgs_array, axis=3)
    return CEUS_data, US_data

def ImageAugmentGenerator(image, mask, batchnum):
    dataGene = ImageDataGenerator(
        rotation_range = 180,
        width_shift_range= 0.1,
        height_shift_range= 0.1,
        zoom_range= 0.1)

    img_generator = dataGene.flow(image, batch_size= batchnum, seed=1)
    mask_generator = dataGene.flow(mask,batch_size= batchnum, seed=1)
    for img,mask in zip(img_generator, mask_generator):
        yield (img,mask)

def ValImageGenerator(image, mask, batchnum):
    dataGene = ImageDataGenerator()
    img_generator = dataGene.flow(image, batch_size= batchnum)
    mask_generator = dataGene.flow(mask,batch_size= batchnum)
    for img,mask in zip(img_generator, mask_generator):
        yield (img,mask)

def display_1st_last(CEUS_images, CEUS_masks, US_images, US_masks):
    # Plot the first and last images
    plt.figure(figsize = (14,6))
    plt.subplot(241)
    plt.imshow(np.squeeze(CEUS_images[0]), cmap = "gray")
    plt.title('First CEUS image')
    plt.subplot(242)
    plt.imshow(np.squeeze(CEUS_masks[0]), cmap = "gray")
    plt.title('First CEUS mask')
    plt.subplot(243)
    plt.imshow(np.squeeze(US_images[0]), cmap = "gray")
    plt.title('First US image')
    plt.subplot(244)
    plt.imshow(np.squeeze(US_masks[0]), cmap = "gray")
    plt.title('First US mask')

    plt.subplot(245)
    plt.imshow(np.squeeze(CEUS_images[-1]), cmap = "gray")
    plt.title('Last CEUS image')
    plt.subplot(246)
    plt.imshow(np.squeeze(CEUS_masks[-1]), cmap = "gray")
    plt.title('Last mask')
    plt.subplot(247)
    plt.imshow(np.squeeze(US_images[-1]), cmap = "gray")
    plt.title('Last US image')
    plt.subplot(248)
    plt.imshow(np.squeeze(US_masks[-1]), cmap = "gray")
    plt.title('Last US image')

    plt.tight_layout()
    plt.show()

def display_img_mask(images, masks, pred_masks, idx):
    plt.figure(figsize = (14,6))
    plt.subplot(131)
    plt.imshow(np.squeeze(images[idx]), cmap = "gray")
    plt.title('original image')
    plt.subplot(132)
    plt.imshow(np.squeeze(masks[idx]), cmap = "gray")
    plt.title('original mask')
    plt.subplot(133)
    plt.imshow(np.squeeze(pred_masks[idx]), cmap = "gray")
    plt.title('preidcted mask')
    plt.tight_layout()
    plt.show()
