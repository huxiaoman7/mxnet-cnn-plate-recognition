#coding=utf-8
""" test.py:对任意一张或一个文件夹的车牌图片做识别预测
    输入:单张图片地址或文件夹地址
    输出:单张图片的准确率或文件夹下所有图片的准确率
"""

__author__ = "Huxiaoman"
__copyright__ = "Copyright (c) 2017 "

import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import  cv2,random
import os
import os.path
from io import BytesIO
#from train import gen_rand, gen_sample

# 所有车牌字符
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ];

# CNN网络模型
def getnet():
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
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc25 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc26 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc27 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24,fc25,fc26,fc27], dim = 0)
    return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")

#预测图片的行准确率和列准确率
def Accuracy_pred(label,pred):
    '''
    预测图片的行准确率和列准确率
    行准确率：如果一个车牌7个字符全部预测正确，行准确率=1，否则=0。
    列准确率：预测正确的车牌字符个数/ 7
    '''
    hit = 0
    total = 1
    # 预测第一个中文字符是否相同
    if label[0].decode('unicode-escape') == pred[0].decode('unicode-escape'):
        hit =1
        ok = True 
    else:
	hit = 0
    for i in range(1,7):
        if label[i] == pred[i]:
	    hit +=1
        else:
	    ok = False
	total +=1
    if hit != total:
        hit_row = 0
    else:
	hit_row = total
    #行准确率
    acc_row = 1.0 * hit_row / total
    #列准确率
    acc_col = 1.0 * hit / total
    print ("预测行准确率为:" + str(acc_row))
    print ("预测列准确率为:" + str(acc_col))
    return acc_row,acc_col

#批量读取测试图片
def batch_read(rootdir):
    pathlist = []
    for parent,dirnames,filenames in os.walk(rootdir):    
       	for filename in filenames:                        
	    filename_path = os.path.join(parent,filename)
	    pathlist.append(filename_path)
    return pathlist

# label的字符串转列表格式
def strtolist(str):
    result=str.split(',')
    head=result[0][1:][1:-1]
    head.decode('unicode-escape')
    bottom=result[len(result)-1][0:-1][2]
    b =[]
    for i in range(1,6):
        c=result[i][2]
        b.append(c)
    b.insert(0,head)
    b.append(bottom)
    return b

#批量预测车牌图片准确率
def TestRecognizeBatch(path):
    # 获取测试图片label
    label_list=[]
    f=open('label.txt','r')
    for line in f.readlines():
	lines = line.split("\n")
	label_list.append(lines)
    f.close()
    label = [strtolist(label_list[i][0]) for i in range(len(label_list))]
    # 获取测试图片路径
    pathlist=[]
    for parent,dirnames,filenames in os.walk(path):
    	filenames.sort(key= lambda x:int(x[:-4]))  
        for filename in filenames:
	    filename_path = os.path.join(parent,filename)
	    pathlist.append(filename_path)
    # 调用模型训练文件
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 30, 120))]
    input_shapes = dict(data_shape)
    sym = getnet()
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])
    # 获取测试图片预测结果predlist
    predlist=[]
    row_list=[]
    col_list=[] 
    for i in range(len(pathlist)):
	print i
	print pathlist[i]
    	img = cv2.imread(pathlist[i])
	img = cv2.resize(img,(120,30))
        img = np.swapaxes(img,0,2)
    	img = np.swapaxes(img,1,2)
	executor.forward(is_train = True, data = mx.nd.array([img]))
	probs = executor.outputs[0].asnumpy()
	line = ''
        pred=[]
        for j in range(probs.shape[0]):
            if j == 0:
                result =  np.argmax(probs[j][0:31])
            if j == 1:
                result =  np.argmax(probs[j][41:65])+41
            if j >  1:
                result =  np.argmax(probs[j][31:65])+31
	    line += chars[result]+" "
	    pred.append(chars[result])
	print ('预测车牌号码为:'+line)
        print label[i]
	#print pred
	predlist.append(pred)
	#Accuracy_pred(label[i],predlist[i])
        row,col= Accuracy_pred(label[i],predlist[i])
	row_list.append(row)
	col_list.append(col)
    #print row_list
    #print col_list
    batch_row_acc=sum(row_list) / len(row_list)
    batch_col_acc=sum(col_list) / len(col_list)
    print ("输入图片数量:" + str(len(row_list)))
    print ("输入图片行准确率:" + str(batch_row_acc))
    print ("输入图片列准确率:" + str(batch_col_acc))
        
    return label,predlist

#预测单张车牌图片的准确率
def TestRecognizeOne(img):
    img = cv2.imread(img)
    img = cv2.resize(img,(120,30))
    cv2.imshow("img",img)
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 30, 120))]
    input_shapes = dict(data_shape)
    sym = getnet()
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])
    executor.forward(is_train = True, data = mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()
    line = ''
    pred=[]
    for i in range(probs.shape[0]):
        if i == 0:
            result =  np.argmax(probs[i][0:31])
        if i == 1:
            result =  np.argmax(probs[i][41:65])+41
        if i >  1:
            result =  np.argmax(probs[i][31:65])+31
	    print result
        line += chars[result]+" "
        pred.append(chars[result])
    print ('predicted: ' + line)
    # 单张图片的label，需要在此处写好
    label = ['川','A','0','9','5','Q','5']
    Accuracy_pred(label,pred)
    cv2.waitKey(0)
    return pred

if __name__ == '__main__':
    # 调用一张图片测试,输入参数为图片路径+名称,如:/Users/shelter/1.png
    #TestRecognizeOne(sys.argv[1])
    # 调用一个文件夹下面的图片做测试,输入参数为图片路径,如:/Users/shelter/picture
    TestRecognizeBatch(sys.argv[1])
