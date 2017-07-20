# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:01:25 2017

@author: lenovo
"""

"""
这个程序是用程序生成网络配置文件，并进行训练
数据直接用ImageData层时看这个网址：
http://www.cnblogs.com/denny402/p/5684431.html
"""

import caffe
from caffe import layers as L,params as P,proto,to_proto

train_lmdb='D:/caffe-master/examples/mnist/lmdb/train_lmdb'#训练集
test_lmdb='D:/caffe-master/examples/mnist/lmdb/test_lmdb'#测试集
train_proto='C:/Users/lenovo/Desktop/mnist/train.prototxt'#训练网络配置
test_proto='C:/Users/lenovo/Desktop/mnist/test.prototxt'#测试网络配置
solver_proto='C:/Users/lenovo/Desktop/mnist/solver.prototxt'#训练参数


#编写一个函数，生成配置文件prototxt
def Lenet(img_list,batch_size,include_acc=False):
    #第一层，数据输入层，以ImageData格式输入--->这个层是直接用图像作为输入
#    data, label = L.ImageData(source=img_list, batch_size=batch_size, ntop=2,root_folder=root,
#        transform_param=dict(scale= 0.00390625))
    data,label=L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=img_list, 
                          transform_param=dict(scale=1./255), #缩放比例
                            ntop=2)
    #第二层：卷积层
    conv1=L.Convolution(data, kernel_size=5, stride=1,num_output=20, pad=0,weight_filler=dict(type='xavier'))
    #池化层
    pool1=L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    #卷积层
    conv2=L.Convolution(pool1, kernel_size=5, stride=1,num_output=50, pad=0,weight_filler=dict(type='xavier'))
    #池化层
    pool2=L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    #全连接层
    fc3=L.InnerProduct(pool2, num_output=500,weight_filler=dict(type='xavier'))
    #激活函数层
    relu3=L.ReLU(fc3, in_place=True)
    #全连接层
    fc4 = L.InnerProduct(relu3, num_output=10,weight_filler=dict(type='xavier'))
    #softmax层
    loss = L.SoftmaxWithLoss(fc4, label)
    
    if include_acc:             # test阶段需要有accuracy层
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)
    
def write_net():
    #写入train.prototxt
    with open(train_proto, 'w') as f:
        f.write(str(Lenet(train_lmdb,batch_size=64)))

    #写入test.prototxt    
    with open(test_proto, 'w') as f:
        f.write(str(Lenet(test_lmdb,batch_size=100, include_acc=True)))

#编写一个函数，生成参数文件
def gen_solver(solver_file,train_net,test_net):
    s=proto.caffe_pb2.SolverParameter()
    s.train_net =train_net
    s.test_net.append(test_net)
    s.test_interval = 1000    #60000/64，测试间隔参数：训练完一次所有的图片，进行一次测试  
    s.test_iter.append(500)  #50000/100 测试迭代次数，需要迭代500次，才完成一次所有数据的测试
    s.max_iter = 10000       #10 epochs , 938*10，最大训练次数
    s.base_lr = 0.01    #基础学习率
    s.momentum = 0.9    #动量
    s.weight_decay = 5e-4  #权值衰减项
    s.lr_policy = 'step'   #学习率变化规则
    s.stepsize=5000         #学习率变化频率
    s.gamma = 0.1          #学习率变化指数
    s.display = 20         #屏幕显示间隔
    s.snapshot = 5000       #保存caffemodel的间隔
    s.snapshot_prefix = 'C:/Users/lenovo/Desktop/mnist/'   #caffemodel前缀
    s.type ='SGD'         #优化算法
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU    #加速
    #写入solver.prototxt
    with open(solver_file, 'w') as f:
        f.write(str(s))
  
#开始训练
def training(solver_proto):
#    caffe.set_device(0)
#    caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()
    
    accuracy = 0
    test_iters = 200#这个值=总测试集数/test_iter=10000/500
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters

    print("Accuracy: {:.3f}".format(accuracy))
#
if __name__ == '__main__':
    write_net()
    gen_solver(solver_proto,train_proto,test_proto)
    print 'Start Train, Please waiting......'
    training(solver_proto)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    