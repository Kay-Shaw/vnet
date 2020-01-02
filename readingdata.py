from skimage import transform
import SimpleITK as sitk
import scipy.ndimage
import nibabel as nib
import numpy as np
import cv2
from PIL import Image

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 255.0 * cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf


def read_data_mhd(shape):
    while True:
        imagedata = np.zeros(shape)
        maskdata = np.zeros(shape)
        for i in range(50):
            if i < 10:
                image_path = r"./data_mhd/TrainingData/Case0" + str(i) + '.mhd'
            else:
                image_path = r"./data_mhd/TrainingData/Case" + str(i) + '.mhd'
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image)
            # print(image.shape)
            image = np.expand_dims(scipy.ndimage.zoom(image, [shape[1] / image.shape[0],
                                                              shape[2] / image.shape[1],
                                                              shape[3] / image.shape[2]],order=1), axis=-1)#原图用双线性插值
            # image = transform.resize(image,shape) #最近邻插值，不如双线性插值表现更好
            imagedata[0] = image
            if i < 10:
                mask_path = r'./data_mhd/TrainingData/Case0' + str(i) + '_segmentation.mhd'
            else:
                mask_path = r'./data_mhd/TrainingData/Case' + str(i) + '_segmentation.mhd'
            mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask)
            # print(mask.shape)
            mask = np.expand_dims(scipy.ndimage.zoom(mask, [shape[1] / mask.shape[0],
                                                              shape[2] / mask.shape[1],
                                                              shape[3] / mask.shape[2]],order=0), axis=-1)#原图用双线性插值

            maskdata[0] = mask



            affine = [[0,0,-1,0],
                      [0,-1,0,0],
                      [-1,0,0,0],
                      [0,0,0,1]]
            print(imagedata.shape)

            im2, cdf = histeq(imagedata)
            # print(im2.shape)
            result = nib.Nifti1Image(np.squeeze(im2),affine)
            nib.save(result, r'E:\论文\vnet\data_nii\his_eq_images\case'+str(i)+'.nii')


            # result_label = nib.Nifti1Image(np.squeeze(mask),affine)
            # nib.save(result_label,r'C:\Users\37474\Desktop\vnet\data_nii\label\caselabel'+str(i)+'.nii')
            # mask = transform.resize(mask, shape,order=0) #标签用最近邻插值,保证resize之后，值还是0和1
            yield imagedata, maskdata

def read_data_train(shape):
    while True:
        image_data = np.zeros(shape)
        mask_data = np.zeros(shape)
        for i in range(30):
            image_path = r'./data_nii/his_eq_images/case'+str(i)+'.nii'
            image = np.expand_dims(nib.load(image_path).get_data(), axis=-1)
            image_data[0] = image
            mask_path = r'./data_nii/label/caselabel'+str(i)+'.nii'
            mask = np.expand_dims(nib.load(mask_path).get_data(), axis=-1)
            mask_data[0] = mask
            yield image_data, mask_data
def read_data_validation(shape):
    while True:
        image_data = np.zeros(shape)
        mask_data = np.zeros(shape)
        for i in range(30,40):
            image_path = r'./data_nii/his_eq_images/case'+str(i)+'.nii'
            image = np.expand_dims(nib.load(image_path).get_data(), axis=-1)
            image_data[0] = image
            mask_path = r'./data_nii/label/caselabel'+str(i)+'.nii'
            mask = np.expand_dims(nib.load(mask_path).get_data(), axis=-1)
            mask_data[0] = mask
            yield image_data, mask_data
def read_data_test(shape):
    while True:
        image_data = np.zeros(shape)
        mask_data = np.zeros(shape)
        for i in range(40,50):
            filename = r'case'+str(i)+'.nii'
            image_path = r'./data_nii/his_eq_images/'+filename
            image = nib.load(image_path)
            images = np.expand_dims(image.get_data(), axis=-1)
            image_data[0] = images
            mask_path = r'./data_nii/label/caselabel'+str(i)+'.nii'
            mask = np.expand_dims(nib.load(mask_path).get_data(), axis=-1)
            mask_data[0] = mask
            yield image_data, mask_data,filename,image.affine

if __name__ == "__main__":
    # for i, j,k,l in load_data([1, 128, 128, 128, 1]):
    # read_data_mhd([1, 128, 128, 128, 1])
    for i, j in read_data_mhd([1, 128, 128, 128, 1]):
        print(i.shape)
        print(j.shape)
        # print(np.squeeze(i))
        # break
