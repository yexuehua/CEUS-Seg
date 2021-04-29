from USModel import *
from USPreprocessing import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from os import listdir
import datetime
import pandas
from pandas import DataFrame

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold

now = datetime.datetime.now

# Directory with images and ground truths
path_imgs = "./Dataset/image"
path_masks = "./Dataset/mask"

test_path_imgs = "./Dataset/test/image"
test_path_masks = "./Dataset/test/mask"

# Introduce parameters
img_row = 768
img_col = 512
img_chan = 3
epochnum = 200
batchnum = 8
input_size = (img_row, img_col, img_chan)

imagesList = listdir(path_imgs)
masksList = [i+"_Merge.nii" for i in imagesList]
print("Number of images:", len(imagesList))

test_imagesList = listdir(test_path_imgs)
test_masksList = [i+"_Merge.nii" for i in test_imagesList]

CEUS_images, US_images = img_load(path_imgs, imagesList)
CEUS_masks, US_masks = img_load(path_masks, masksList)

test_CEUS_images, test_US_images = img_load(test_path_imgs, test_imagesList)
test_CEUS_masks, test_US_masks = img_load(test_path_masks, test_masksList)

print("Images shape: (1)CEUS_image ", CEUS_images.shape, "(2)US_image ", US_images.shape)
print("mask shape: (1)CEUS_mask ", CEUS_masks.shape, "(2)US_mask ", US_masks.shape)

# Plot the first and last images
#display_1st_last(CEUS_images, CEUS_masks, US_images, US_masks)

# Evaluate the models using using k-fold cross-validation
n_folds = 5

numepochs = np.zeros(n_folds,)
dice_score = np.zeros(n_folds,)
iou_score = np.zeros_like(dice_score)
rec_score = np.zeros_like(dice_score)
prec_score = np.zeros_like(dice_score)
globacc_score = np.zeros_like(dice_score)
auc_roc_score = np.zeros_like(dice_score)

# Prepare cross-validation
kfold = KFold(n_folds, shuffle=True, random_state=1)

kfold_train_idx = []
kfold_valid_idx = []
valid_imgs_list = []

for train_ix, valid_ix in tqdm(kfold.split(CEUS_images)):
    kfold_train_idx.append(train_ix)
    kfold_valid_idx.append(valid_ix)
    valid_imgs_list.append(np.array(imagesList)[valid_ix])

# df_valid_image = pandas.DataFrame(np.array(valid_imgs_list))
# df_valid_image['index'] = np.array(kfold_valid_idx)
# df_valid_image.to_csv("df_valid_image.csv")

run = 0;
# enumerate splits
for k in range(n_folds):

    # Display the run number
    print('Run #', run+1)

    # Define  the model
    model = Network(input_size)

    print(kfold_train_idx[k], "\n", kfold_valid_idx[k])

    # Split into train and valid sets
    imgs_train, masks_train, imgs_valid, masks_valid = CEUS_images[kfold_train_idx[k]], CEUS_masks[kfold_train_idx[k]], \
                                                     CEUS_images[kfold_valid_idx[k]],CEUS_masks[kfold_valid_idx[k]]

    TrainGene = ImageAugmentGenerator(imgs_train, masks_train, batchnum)
    for showimg,showmask in TrainGene:
        showimg,showmask = next(TrainGene)
        for i in range(batchnum):
            display_img_mask(showimg,showmask,showmask,i)
    ValidGene = ValImageGenerator(imgs_valid, masks_valid, batchnum)

    # showimg,showmask = next(TrainGene)
    # showvalimg,showvalmask = next(ValidGene)
    # display_img_mask(showimg,showmask,showmask,1)
    # display_img_mask(showvalimg,showvalmask,showvalmask,1)

    # Compile and fit the  model
    model.compile(optimizer = Adam(lr = 0.0001), loss = dice_loss, metrics = [dsc])
    if not os.path.exists('./ModelCheckpoint/KFold'+str(run)):
        os.mkdir('./ModelCheckpoint/KFold'+str(run))
        os.mkdir('./ModelCheckpoint/KFold' + str(run) + "/checkpoint")
    model_checkpoint = ModelCheckpoint('./ModelCheckpoint/KFold'+str(run)+'/checkpoint/wights.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True)
    t = now()
    callbacks = [model_checkpoint]#EarlyStopping(monitor='val_loss', patience = 20),
    #history = model.fit(imgs_train, masks_train, validation_split=0.15, batch_size=batchnum,epochs=epochnum, verbose=2, callbacks=callbacks)

    history = model.fit_generator(TrainGene, steps_per_epoch=len(kfold_train_idx[k])//batchnum, epochs=epochnum, validation_data=ValidGene, validation_steps=np.ceil(len(kfold_valid_idx[k])/batchnum))
    print('Training time: %s' % (now() - t))

    # Plot the loss and accuracy
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['dsc']
    val_acc = history.history['val_dsc']

    epochsn = np.arange(1, len(train_loss)+1,1)
    plt.figure(figsize = (12,5))
    plt.subplot(121)
    plt.plot(epochsn,train_loss, 'b', label='Training Loss')
    plt.plot(epochsn,val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('LOSS, Epochs={}, Batch={}'.format(epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(122)
    plt.plot(epochsn, acc, 'b', label='Training Dice Coefficient')
    plt.plot(epochsn, val_acc, 'r', label='Validation Dice Coefficient')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('DSC, Epochs={}, Batch={}'.format(epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('CSC')
    plt.savefig("kfold"+str(run)+".png")


    # Make predictions
    t = now()
    preds = model.predict(test_CEUS_images)
    print('Testing time: %s' % (now() - t))

    # Evaluate model
    num_test = len(test_imagesList)
    # Calculate performance metrics
    dsc_sc = np.zeros((num_test,1))
    iou_sc = np.zeros_like(dsc_sc)
    rec_sc = np.zeros_like(dsc_sc)
    tn_sc = np.zeros_like(dsc_sc)
    prec_sc = np.zeros_like(dsc_sc)
    thresh = 0.5
    for i in range(num_test):
        dsc_sc[i], iou_sc[i], rec_sc[i], prec_sc[i] = auc(test_CEUS_masks[i], preds[i] >thresh)
    print('-'*30)
    print('USING THRESHOLD', thresh)
    print('\n DSC \t\t{0:^.3f} \n IOU \t\t{1:^.3f} \n Recall \t{2:^.3f} \n Precision\t{3:^.3f}'.format(
            np.sum(dsc_sc)/num_test,
            np.sum(iou_sc)/num_test,
            np.sum(rec_sc)/num_test,
            np.sum(prec_sc)/num_test ))

    '''
    # To plot a set of images with predicted masks uncomment these lines
    num_disp = 10
    j=1
    plt.figure(figsize = (14,num_disp*3))
    for i in range(num_disp):
        plt.subplot(num_disp,4,j)
        plt.imshow(np.squeeze(imgs_test[i]), cmap='gray')
        plt.title('Image')
        j +=1
        plt.subplot(num_disp,4,j)
        plt.imshow(np.squeeze(masks_test[i]),cmap='gray')
        plt.title('Mask')
        j +=1
        plt.subplot(num_disp,4,j)
        plt.imshow(np.squeeze(preds[i]))
        plt.title('Prediction')
        j +=1
        plt.subplot(num_disp,4,j)
        plt.imshow(np.squeeze(np.round(preds[i])), cmap='gray')
        plt.title('Rounded; IOU=%0.2f, Rec=%0.2f, Prec=%0.2f' %(iou_sc[i], rec_sc[i], prec_sc[i]))
        j +=1
    plt.tight_layout()
    plt.show()    
    '''

    # Confusion matrix
    confusion = confusion_matrix( test_CEUS_masks.ravel(),preds.ravel()>thresh)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print(' Global Acc \t{0:^.3f}'.format(accuracy))

    # Area under the ROC curve
    AUC_ROC = roc_auc_score(preds.ravel()>thresh, test_CEUS_masks.ravel())
    print(' AUC ROC \t{0:^.3f}'.format(AUC_ROC))
    print('\n')
    print('*'*60)

    # Save outputs
    numepochs[run] = epochsn[-1]
    dice_score[run] = np.sum(dsc_sc)/num_test
    iou_score[run] = np.sum(iou_sc)/num_test
    rec_score[run] = np.sum(rec_sc)/num_test
    prec_score[run] = np.sum(prec_sc)/num_test
    globacc_score[run] = accuracy
    auc_roc_score[run] = AUC_ROC
    run +=1


# Display the scores in a table

df = DataFrame({'Epochs Number': numepochs, 'Dice Score': dice_score, 'IOU Score': iou_score, 'Recall (Sensitivity)': rec_score, 'Precision': prec_score, 'Global Accuracy': globacc_score, 'AUC-ROC': auc_roc_score})
df.to_csv("df.csv")
# Calculate mean values of the scores
numepochs_mean = np.mean(numepochs)
dice_mean = np.mean(dice_score)
iou_mean = np.mean(iou_score)
rec_mean = np.mean(rec_score)
prec_mean = np.mean(prec_score)
globacc_mean = np.mean(globacc_score)
auc_roc_mean = np.mean(auc_roc_score)

# Mean values of the scores
df2 = DataFrame({'Epochs Number Mean': numepochs_mean, 'Dice Score Mean': dice_mean, 'IOU Score Mean': iou_mean, 'Recall (Sensitivity) Mean': rec_mean, 'Precision Mean': prec_mean, 'Global Accuracy Mean': globacc_mean, 'AUC-ROC Mean': auc_roc_mean},index=[5])
df2.to_csv("df2.csv")

# Calculate standard deviations of the scores
dice_std = np.std(dice_score)
iou_std = np.std(iou_score)
rec_std = np.std(rec_score)
prec_std = np.std(prec_score)
globacc_std = np.std(globacc_score)
auc_roc_std = np.std(auc_roc_score)


# Standard deviations of the scores
df3 = DataFrame({'Dice Score STD': dice_std, 'IOU Score STD': iou_std, 'Recall (Sensitivity) STD': rec_std, 'Precision STD': prec_std, 'Global Accuracy STD': globacc_std, 'AUC-ROC STD': auc_roc_std}, index=[5])
df3.to_csv("df3.csv")
