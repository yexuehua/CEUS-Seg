import os
import numpy as np
import SimpleITK as sitk

# Load the images
def img_load(dir_path, imgs_list):
    #inital the list
    CEUS_data_list = []
    US_data_list = []
    for i in range(len(imgs_list)):
        # read dicom data
        print(os.path.join(dir_path, imgs_list[i]))
        itkimg_f1 = sitk.ReadImage(os.path.join(dir_path, imgs_list[i]))
        f1_data = sitk.GetArrayFromImage(itkimg_f1)
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
            CEUSImage = f1_data[0][:, 0:w // 2]
            USImage = f1_data[0][:, w // 2:]
            CEUSImage = np.expand_dims(CEUSImage, axis=2)
            USImage = np.expand_dims(USImage, axis=2)
        # store the data to a list
        CEUS_data_list.append(CEUSImage)
        US_data_list.append(USImage)

    CEUS_data = np.array(list(CEUS_data_list))
    US_data = np.array(list(CEUS_data_list))
    # Expand the dimensions of the arrays
    # imgs_array = np.expand_dims(imgs_array, axis=3)
    return CEUS_data, US_data