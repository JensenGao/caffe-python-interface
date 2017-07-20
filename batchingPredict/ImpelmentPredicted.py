# -*- coding: utf-8 -*-


'''
批量预测图片，并将结果保存至.txt文件中
'''


import os,glob
import batchPredicted
import caffe
caffe.set_mode_cpu()#cpu模式

dir_path='D:/caffe-master/examples/cifar10/img/'#图像路径
extensions=['jpg']#文件扩展名
os.chdir(dir_path)

imageList=[]

for extension in extensions:
    extension='*.'+extension
    imageList=[os.path.realpath(e) for e in glob.glob(extension)]
                             
caffe_model='D:/caffe-master/examples/cifar10/caffemodel/_iter_4000.caffemodel'
depoly='D:/caffe-master/examples/cifar10/net/cifar10_quick.prototxt'
#mean_file='D:/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy'

#自己的数据的均值图像文件mean.binaryproto
mean_file='D:/caffe-master/examples/cifar10/mean_file/mean.binaryproto'
labels='D:/caffe-master/examples/cifar10/data/cifar-10-batches-bin/batches.meta.txt'
class_file='D:/caffe-master/examples/cifar10/bat/result.txt'#预测后的结果保存在这个文件中
f=open(class_file,'w')

for file in imageList:
    image=file
    order,predictedProb,predictedClass=batchPredicted.batchClassifyImage(caffe_model,
                                                                   depoly,
                                                                   mean_file,
                                                                   labels,
                                                                   image)
    print image,predictedProb,predictedClass
    f.write(imageList[order]+'\t'+str(predictedProb)+'   '+predictedClass+'\n')
f.close()
