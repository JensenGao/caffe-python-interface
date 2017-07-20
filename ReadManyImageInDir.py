# -*- coding: utf-8 -*-


"""
读取一个文件夹下的图片--->批量处理
"""


import os,glob
import matplotlib.pyplot as plt
from PIL import Image

dir_path='D:/caffe-master/examples/cifar10/img/'
extensionName=['jpg']#扩展名
os.chdir(dir_path)
imageList=[]

for extension in extensionName:
    extension='*.'+extension
    imageList+=[os.path.realpath(e) for e in glob.glob(extension)]
     
f=open('C:/Users/lenovo/Desktop/image.txt','w')
for file in imageList:
    print file
    f.write(file+'\n')#写入txt文件
    plt.figure()#每次在一个新图中画图
    plt.imshow(Image.open(file))#打开每个图像并显示
    plt.axis('off')
