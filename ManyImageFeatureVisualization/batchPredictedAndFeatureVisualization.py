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

    '''
    上面的得到分类的结果，下面开始显示网络各层的详细信息以及
    各层特征及权重的可视化
    '''
    #每层输出的大小###输出形式：层名 （batch_size,channel_dim,height,width）
    for layer_name,blob in net.blobs.iteritems():
        print layer_name+'\t'+str(blob.data.shape)
    print '\n'
    #每层输入输出的大小：batch,channel,height,width,param[0]-权重；param[1]-偏置
    for layer_name,param in net.params.iteritems():
        print layer_name+'\t'+str(param[0].data.shape),str(param[1].data.shape)
        '''
        下面开始权重和特征的可视化
        '''
    #查看各层的参数
    print 'params visualization'
    filters=net.params['conv1'][0].data###conv1层的参数可视化
    vis_square(filters.transpose(0,2,3,1))
    plt.figure()#新图
    #特征图的可视化(特征图可视化时要注意显示的个数  滤波器等的个数)

    print 'output visualization'
    feat=net.blobs['conv3'].data[0]#convert输出的特征图(10为显示个数，与滤波器个数对应)
    vis_square(feat)
    plt.figure()

    feat=net.blobs['pool3'].data[0,:32]#convert输出的特征图(10为显示个数，与滤波器个数对应)
    vis_square(feat)
    plt.figure()

    feat = net.blobs['pool3'].data[0]#全部显示
    vis_square(feat)
    plt.figure()

    feat = net.blobs['data'].data[0,:3]
    vis_square(feat)
    plt.figure()
    '''
    上面是每个特征图画一个图里
    卷积核池化层的特征可以直接显示；全连接层和prob不能直接显示
    而用下面的方法显示
    '''


    feat=net.blobs['ip1'].data[0]
    plt.subplot(2,1,1)
    plt.plot(feat.flat)#输出值
    plt.subplot(2,1,2)
    _=plt.hist(feat.flat[feat.flat>0],bins=100)#直方图

    #概率输出
    feat=net.blobs['prob'].data[0]
    plt.figure(figsize=(15,3))
    plt.plot(feat.flat)


    return predictedProb,predictedClass#返回概率和类别 
#可视化辅助函数   
def vis_square(data):
    #对要显示的内容归一化
    data=(data-data.min())/(data.max()-data.min())
    
    #将显示形式做成一个方形
    n=int(np.ceil(np.sqrt(data.shape[0])))
    padding=(((0,n**2-data.shape[0]),
              (0,1),(0,1))   #在显示的方格之间加一个空间
              +((0,0),)*(data.ndim-3))#最后一维不填充
    data=np.pad(data,padding,mode='constant',constant_values=1)#用1填充（白色）
    #将滤波器嵌入图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.axis('off')
