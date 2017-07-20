# -*- coding: utf-8 -*-


"""
batch predict some image!
"""

import caffe 
import numpy as np
import matplotlib.pyplot as plt

def batchClassifyImage(caffe_model,depoly,mean_file,labels_file,image_path):#分类函数
    net=caffe.Net(depoly,caffe_model,caffe.TEST)#构造网络
    transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    #transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))#均值问津为ilsvrc_2012_mean.npy时一条语句就行
    
    #这里用自己的均值图像文件mean.binaryproto
    mean_blob=caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_file,'rb').read())
    mean_npy=caffe.io.blobproto_to_array(mean_blob)
    b2=mean_npy[0,:,:,:]
    transformer.set_mean('data',b2.mean(1).mean(1))
    
    
    transformer.set_raw_scale('data',255)
    transformer.set_channel_swap('data',(2,1,0))
    
    image=caffe.io.load_image(image_path)
    plt.imshow(image)
    plt.axis('off')#关闭坐标轴显示
    
    net.blobs['data'].reshape(1,3,32,32)
    net.blobs['data'].data[...]=transformer.preprocess('data',image)#转换成为caffe的输入形式、
    
    labels=np.loadtxt(labels_file,str,delimiter='\t')#类别
    #print labels#打印出来看一下
    
    output=net.forward()
    prob=output['prob'][0]#预测数来的概率
    
    class_order=prob.argsort()[-1]#取类别的序号
    
    predictedClass=labels[class_order]#真正的类别
    predictedProb=prob[class_order]

    return class_order,predictedProb,predictedClass#返回概率和类别 
