#coding=utf-8
"""
   train.py:训练CNN模型
"""

__author__ = "Huxiaoman"
__copyright__ = "Copyright (c) 2017 "

import sys
sys.path.insert(0, "../../python")
sys.setrecursionlimit(1000000)
import logging
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO
from genPlate import *
import scipy.misc
import math
import mxnet.misc
import multiprocessing

class OneBatch(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


#多进程操作读取数据
#class DataIter(mx.io.DataIter):
#    def __init__(self,count,batch_size,num_label,height,width):
#        super(DataIter,self).__init__()
#        self.batch_size = batch_size
#        self.cursor = -batch_size
#        self.num_data = count
#        self.height = height
#        self.width = width
#        self.provide_data = [('data',(batch_size,3,height,width))]
#        self.provide_label = [('softmax_label',(self.batch_size,num_label))]
#        # 首先定义一个Queue对象，用来存储 Prefetch 的数据
#	self.q = multiprocessing.Queue(maxsize = 2)
#        # 创建4个读取数据的进程，具体的读取方法在self.write中
#        self.pws = [multiprocessing.Process(target = self.write) for i in range(4)]
#        for pw in self.pws:
#	    pw.daemon = True
#	    pw.start()
#
#    def write(self):
#        while True:
#	    for i in range(self.batch_size):
#	        # 获取一张图片,numpy.array 格式
#                one_img = func_get_img_data()
#		# 获取该图片的label数据, numpy.array格式
#		one_label = func_get_label_data()
#		data.append(one_img)
#		label.append(one_label)
#            data_all = [mx.nd.array(data)]
#	    label_all = [mx.nd.array(label)]
#	    # 创建一个 Batch 的数据
#	    data_batch = OneBatch(data_all,label_all)
#	    # block = True,允许在队列满的时候阻塞，timeout = None, 永不超时
#	    self.q.put(obj = data_batch,block = True,timeout = None)
#    def iter_next(self):
#	if self.q.empty():
#	    logging.debug("Waiting for data")
#	if self.iter_next():
#	    return self.q.get(block = True,timeout = None)
#	else:
#	    raise StopIteration
#
#    def reset(self):
#	self.cursor = -self.batch_size + (self.cursor%self.num_data) % self.batch_size

class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def rand_range(lo,hi):
    return lo+r(hi-lo);


def gen_rand():
    name = "";
    label= [];
    label.append(rand_range(0,31));
    label.append(rand_range(41,65));
    for i in xrange(5):
        label.append(rand_range(31,65))

    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in xrange(5):
        name+=chars[label[i+2]]
    return name,label


def gen_sample(genplate, width, height):
    num,label = gen_rand()
    img = genplate.generate(num)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return label, img

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        print "start"
    def __iter__(self):

        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.genplate, self.width, self.height)
                data.append(img)
                label.append(num)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_ocrnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")
    
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu2)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 120)
    fc21_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc21 = mx.symbol.FullyConnected(data = fc21_dropout, num_hidden = 65)

    fc22_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc22 = mx.symbol.FullyConnected(data = fc22_dropout, num_hidden = 65)

    fc23_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc23 = mx.symbol.FullyConnected(data = fc23_dropout, num_hidden = 65)

    fc24_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc24 = mx.symbol.FullyConnected(data = fc24_dropout, num_hidden = 65)

    fc25_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc25 = mx.symbol.FullyConnected(data = fc25_dropout, num_hidden = 65)

    fc26_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc26 = mx.symbol.FullyConnected(data = fc26_dropout, num_hidden = 65)

    fc27_dropout = mx.symbol.Dropout(fc1,p=0.5)
    fc27 = mx.symbol.FullyConnected(data = fc27_dropout, num_hidden = 65)

    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24,fc25,fc26,fc27], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 7):
        ok = True
        for j in range(7):
            k = i * 7 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

#增加自适应的学习率模块
#def _get_lr_scheduler(args, kv):
#    if 'lr_factor' not in args or args.lr_factor >= 1:        
#        return (args.lr, None)    
#    epoch_size = args.num_examples / args.batch_size    
#    if 'dist' in args.kv_store:        
#        epoch_size /= kv.num_workers    
#    begin_epoch = args.load_epoch if args.load_epoch else 0    
#    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]    
#    lr = args.lr    
#    for s in step_epochs:        
#        if begin_epoch >= s:            
#            lr *= args.lr_factor    
#    if lr != args.lr:        
#        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))    
#
#    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]    
#    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))
#
#def train():
#    network = get_ocrnet()
#    batch_size = 8
#    num_epoch = 15
#    num_gpus = 2
#    datadir='/Users/shelter/mxnet-cnn-plate-recognition'
#    head = '%(asctime)-15s %(message)s'
#    logging.basicConfig(level =logging.DEBUG,format=head)
#    devs = [mx.cpu(i) for i in range(num_gpus)]
#    model = mx.model.FeedForward(ctx=devs, #使用GPU来跑
#                                 symbol = network,
#                                 num_epoch = num_epoch,
#                                 #optimizer = 'adam',
#                                 learning_rate = 0.5,
#                                 lr_scheduler=mx.misc.FactorScheduler(step=5),
#                                 wd = 0.00001,
#                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
#                                 momentum = 0.9)
#
#    train_dataiter = mx.io.ImageRecordIter(
#            path_imgrec=datadir+"/train_train.rec",
#            #mean_img=datadir+"/train_train.bin",
#            rand_crop=True,
#            rand_mirror=True,
#            data_shape=(3,30,120),
#            batch_size=batch_size,
#            preprocess_threads=4)
#    test_dataiter = mx.io.ImageRecordIter(
#            path_imgrec=datadir+"/train_val.rec",
#            #mean_img=datadir+"/train_val.bin",
#            rand_crop=False,
#            rand_mirror=False,
#            data_shape=(3,30,120),
#            batch_size=batch_size,
#            preprocess_threads=4)
#
#    model.fit(X = train_dataiter, eval_data = test_dataiter, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 50))
#    model.save("/data/mxnet/cnn-ocr-02")
#    print gen_rand()
#

def train():
    network = get_ocrnet()
    devs = [mx.cpu(i) for i in range(2)]
    model = mx.model.FeedForward(ctx=devs, #使用GPU来跑
                                 symbol = network,
                                 num_epoch = 15,
				 #optimizer = 'adam',
                                 learning_rate = 0.5,
                                 #lr_scheduler=mx.misc.FactorScheduler(step=5), 
				 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
			         momentum = 0.9)
    batch_size = 8
    data_train = OCRIter(100, batch_size, 7, 30, 120)
    data_test = OCRIter(100, batch_size,7, 30, 120)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 50))
    model.save("/data/mxnet/cnn-ocr-model")
    print gen_rand()


if __name__ == '__main__':
    train();
