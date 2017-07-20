# -*- coding: utf-8 -*-

#import sys   
#reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入   
#sys.setdefaultencoding('utf-8')   
#  
#str = '中文'   
#str.encode('gb18030')  

import numpy as np
import matplotlib.pyplot as plt
import caffe

caffe.set_mode_cpu()

model_deploy='D:/caffe-master/examples/cifar10/net/cifar10_quick.prototxt'
model_weights='D:/caffe-master/examples/cifar10/caffemodel/_iter_4000.caffemodel'
net=caffe.Net(model_deploy,model_weights,caffe.TEST)#定义网络模型，并测试

#image=caffe.io.load_image('D:/caffe-master/examples/images/cat.jpg')#载入图像
image=caffe.io.load_image('D:/caffe-master/examples/cifar10/img/1.jpg')
print "show image!"
plt.figure(),plt.title('Origin')
plt.imshow(image)#显示

###下面转换图像的格式，适合caffe输入
transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})#设定图片的格式
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))#RGB2BGR
transformed_image=transformer.preprocess('data',image)#转换图像

#ImageNet2012的均值文件，
#mean_image=np.load('D:/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy')#均值图像,这里np.load()必须要
#transformer.set_mean('data',mean_image.mean(1).mean(1))
#print 'mean_subtracted values:',zip('BGR',mean_image)

#自己的均值图像
mean_image='D:/caffe-master/examples/cifar10/mean_file/mean.binaryproto'#读入自己的均值图像，这里不用np.load(),直接给出路径就行
#下面将.binaryproto文件转换为.npy的形式
mean_blob=caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_image,'rb').read())
mean_npy=caffe.io.blobproto_to_array(mean_blob)
b2=mean_npy[0,:,:,:]#转换到.npy形式了，下面直接进行操作
transformer.set_mean('data',b2.mean(1).mean(1))
print 'mean_subtracted values:',zip('BGR',b2)

#下面显示减去均值后的图像
input_data=net.blobs['data'].data
plt.figure(),plt.title('Subtract mean')
plt.imshow(transformer.deprocess('data',input_data[0]))


#下面设置进行分类的参数
net.blobs['data'].reshape(1,3,32,32)#根据网络resahpe

net.blobs['data'].data[...]=transformed_image#图像复制到网络中
#分类
output=net.forward()
output_prob=output['prob'][0]#batch第一个图像
print "predicted class is:",output_prob.argmax()#输出概率最大的类别；这里打印对应的数字类别

#下面验证标签
labels_file='D:/caffe-master/examples/cifar10/data/cifar-10-batches-bin/batches.meta.txt'
labels=np.loadtxt(labels_file,str,delimiter='\t')

top_inds=output_prob.argsort()[-1]
#argsort升序排列，用[-1]取倒数第一个（最大值）
#若要取top5，则为output_prob.argsort()[::-1][:5]
print "Probalities is {0}'\n'Label is {1}".format(output_prob[top_inds],labels[top_inds])
