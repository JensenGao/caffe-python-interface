# -*- coding: utf-8 -*-


import time
start_time=time.clock()#开始时间

from pylab import *
import matplotlib.pyplot as plt
import caffe
caffe.set_mode_cpu()#cpu模式
#caffe.set_device(0)#gpu模式
#caffe.set_mode_gpu()

solver=caffe.SGDSolver('D:/caffe-master/examples/cifar10/net/cifar10_quick_solver.prototxt')
print solver
###下面是训练的一些参数
niter=4000#迭代次数
display_iter=400
test_iter=24#测试迭代次数
test_interval=200#每隔200次测试

#train loss
train_loss=zeros(ceil(niter*1.0/display_iter))
#test loss
test_loss=zeros(ceil(niter*1.0/test_interval))
#test accuracy
test_acc=zeros(ceil(niter*1.0/test_interval))
#iteration 0
solver.step(1)#只迭代一个batch，进行前向传播、反向传播和更新权重
_train_loss=0;_test_loss=0;_accuracy=0
for it in range(niter):
    solver.step(1)
    _train_loss+=solver.net.blobs['loss'].data
    if it%display_iter==0:#显示
        train_loss[it//display_iter]=_train_loss/display_iter
        _train_loss=0
    if it%test_interval==0:#测试
        for test_it in range(test_iter):
            solver.test_nets[0].forward()#前向传播
            _test_loss+=solver.test_nets[0].blobs['loss'].data
            _accuracy+=solver.test_nets[0].blobs['accuracy'].data
        test_loss[it/test_interval]=_test_loss/test_iter
        test_acc[it/test_interval]=_accuracy/test_iter
        _test_loss=0
        _accuracy=0
print '\nPlot the train loss and test loss and test accuracy!\n'

_,ax1=plt.subplots();
ax2=ax1.twinx()

ax1.plot(display_iter*arange(len(train_loss)),train_loss,'g')
ax1.plot(test_interval*arange(len(test_loss)),test_loss,'y')
ax2.plot(test_interval*arange(len(test_acc)),test_acc,'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.show()

end_time=time.clock()-start_time#共消耗的时间
print end_time,'s'


'''
solver.step(1):只迭代一个batch，进行前向传播、反向传播和更新权重
solver.forward():只进行前向传播
'''         
