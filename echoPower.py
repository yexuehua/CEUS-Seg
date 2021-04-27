import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os

"""
get data matrix through simpleitk
"""

# read single image
path = r"D:\yexuehua\MyProject\DCEUS\data"
file_list = os.listdir(path)
img_head1 = sitk.ReadImage(os.path.join(path, file_list[1]))
img_head2 = sitk.ReadImage("IM000001")
img_head3 = pydicom.read("IM000001")
img_array = sitk.GetArrayFromImage(img_head1)

plt.imshow(np.squeeze(img_array,0))
plt.show()

"""
convert the dicom data to echo-power
"""
