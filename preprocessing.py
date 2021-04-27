import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Directory with images and ground truths
path_imgs = r"D:\python\CEUS-python\Dataset\image"
path_masks = r"D:\python\CEUS-python\Dataset\mask"

imagesList = listdir(path_imgs)
masksList = [i+"_merge" for i in imagesList]

num_imgs = len(imagesList)
print("Number of images:", num_imgs)

CEUS_data_list = []
US_data_list = []

# Load the images
def img_load(dir_path, imgs_list, imgs_array):
    for i in range(num_imgs):
        # read dicom data
        itkimg_f1 = sitk.ReadImage(os.path.join(dir_path, imgs_list[i]))
        f1_data = sitk.GetArrayFromImage(itkimg_f1)
        # get the shape data
        _,h,w,c = f1_data.shape
        CEUSImage = np.zeros((h, w/2, c))
        USImage = np.zeros_like(CEUSImage)

        CEUSImage = f1_data[0][:, 0:w/2, :]
        USImage = f1_data[0][:, w/2:, :]

        CEUS_data_list.append(list(CEUSImage))
        US_data_list.append(list(USImage))

    CEUS_data_list = np.array(CEUS_data_list)
    US_data_list = np.array(US_data_list)
    print(US_data_list.shape)
    # Expand the dimensions of the arrays
    # imgs_array = np.expand_dims(imgs_array, axis=3)
    return US_data_list

imgs = img_load(path_imgs, imagesList, imgs)
masks = img_load(path_masks, masksList, masks)





# img_files = os.listdir(img_path)
# mask_files = os.listdir(mask_path)
#
# itkimg_f1 = sitk.ReadImage(os.path.join(img_path,img_files[1]))
# f1_data = sitk.GetArrayFromImage(itkimg_f1)
# print(f1_data.shape)
#
# itkmask_f1 = sitk.ReadImage(os.path.join(mask_path,mask_files[0]))
# f1_mask_data = sitk.GetArrayFromImage(itkmask_f1)
# print(f1_mask_data.shape)
# print(np.max(f1_data))
# for img in img_files:
#
# grayimage = np.zeros(f1_data.shape)
# w1 = 299
# w2 = 587
# w3 = 114
#
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
# # f1_data[0][:,:,0] = clahe.apply(f1_data[0][:,:,0])
# # f1_data[0][:,:,1] = clahe.apply(f1_data[0][:,:,1])
# # f1_data[0][:,:,2] = clahe.apply(f1_data[0][:,:,2])
# grayimage = (w1*f1_data[0][:,:,0] + w2*f1_data[0][:,:,1] + w3*f1_data[0][:,:,2])/1000
# print(np.max(grayimage))
#
# grayimage = grayimage.astype('uint8')
# print(np.max(grayimage))
thresh = cv2.Canny(f1_data[0][:,:,0],128,256)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_approx_hull_polygon(img, cnts):

    #
    area = []
    #
    # for k in range(len(contours)):
    #     area.append(cv2.contourArea(cnts[k]))
    #
    # max_id = np.argmax(np.array(area))
    # cv2.drawContours(img, cnts, max_id, (255, 0, 0), 2)


    # img = np.copy(img)
    # img = np.zeros(img.shape, dtype=np.uint8)

    # cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
    #
    # epsilion = img.shape[0]/32
    # approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    # cv2.polylines(img, approxes, True, (0, 255, 0), 2)  # green
    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    #cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red
    for k in range(len(contours)):
        area.append(cv2.contourArea(hulls[k]))

    max_id = np.argmax(np.array(area))
    cv2.drawContours(img, hulls, max_id, (0, 0, 255), 2)
    new_mask = np.zeros(img.shape)
    cv2.minEnclosingCircle(cnts[max_id])
    cv2.fillConvexPoly(new_mask,hulls[max_id], (0,0,1))

    # 我个人比较喜欢用上面的列表解析，我不喜欢用for循环，看不惯的，就注释上面的代码，启用下面的
    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.01 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    return img, new_mask
img, newma = draw_approx_hull_polygon(f1_data[0], contours)

# print(thresh
plt.figure("Image")
plt.imshow(img)
plt.show()
