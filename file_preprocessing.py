import os
import shutil

path = r"G:\ye\Postgraduate\MyCode\Python-Project\CEUS-python\dataset\alldata"
img_path = r"G:\ye\Postgraduate\MyCode\Python-Project\CEUS-python\dataset\image"
mask_path = r"G:\ye\Postgraduate\MyCode\Python-Project\CEUS-python\dataset\mask"
patient = os.listdir(path)
for p in patient:
    files = os.listdir(os.path.join(path,p))
    for f in files:
        if (f[-3:]=="nii"):
            shutil.move(os.path.join(path,p,f), os.path.join(mask_path,p+f))
        else:
            shutil.move(os.path.join(path,p,f), os.path.join(img_path,p+f))
