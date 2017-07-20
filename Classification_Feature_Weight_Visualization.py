# -*- coding: utf-8 -*-

'''
分类的时候可视化特征
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import caffe

#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
start=time.clock()#开始的时间

model_deploy='E:/AMyProject/RoadClassification/TrainCaffeNetModel/Model/deploy.prototxt'
model_weights='E:/AMyProject/RoadClassification/TrainCaffeNetModel/Model/_iter_1000.caffemodel'
net=caffe.Net(model_deploy,model_weights,caffe.TEST)#定义网络模型，并测试

image=caffe.io.load_image('E:/AMyProject/RoadClassification/VideoData/images30frames/0m_019.jpg')#载入图像
#image=caffe.io.load_image('D:/caffe-master/examples/cifar10/img/1.jpg')
print "show image!"
plt.imshow(image)#显示
plt.figure()

###下面转换图像的格式，适合caffe输入
transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})#设定图片的格式
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))#RGB2BGR
transformed_image=transformer.preprocess('data',image)#转换图像

#自己的均值图像
mean_image='E:/AMyProject/RoadClassification/TrainCaffeNetModel/imagenet_mean.binaryproto'#读入自己的均值图像，这里不用np.load(),直接给出路径就行
#下面将.binaryproto文件转换为.npy的形式
mean_blob=caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_image,'rb').read())
mean_npy=caffe.io.blobproto_to_array(mean_blob)
b2=mean_npy[0,:,:,:]
transformer.set_mean('data',b2.mean(1).mean(1))
#print 'mean_subtracted values:',zip('BGR',b2)#打印均值信息

#下面设置进行分类的参数
net.blobs['data'].reshape(10,3,32,32)#根据网络resahpe

net.blobs['data'].data[...]=transformed_image#图像复制到网络中
#分类
output=net.forward()
output_prob=output['prob'][0]#batch第一个图像
print "predicted class is:",output_prob.argmax()#输出概率最大的类别；这里打印对应的数字类别

#下面验证标签
labels_file='E:/AMyProject/RoadClassification/VideoData/DataSet/labels.txt'
labels=np.loadtxt(labels_file,str,delimiter='\t')

top_inds=output_prob.argsort()[-1]
#argsort升序排列，用[-1]取倒数第一个（最大值）
#若要取top5，则为output_prob.argsort()[::-1][:5]
print "Probalities is {0}'\n'Label is {1}".format(output_prob[top_inds],labels[top_inds])

cost_time=time.clock()-start
print 'It\'s take {0} seconds'.format(cost_time)

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
a=plt.hist(feat.flat[feat.flat>0],bins=100)#直方图

#概率输出
feat=net.blobs['prob'].data[0]
plt.figure(figsize=(15,3))
plt.plot(feat.flat)

