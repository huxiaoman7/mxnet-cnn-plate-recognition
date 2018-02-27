# 基于CNN的OCR车牌识别

------

- 本项目地址：https://github.com/huxiaoman7/mxnet-cnn-plate-recognition.git
- 车牌识别包含两部分：车牌检测和车牌识别。本模型主要用于车牌的识别。


------
### 所需环境

> * Python2.7
> * Mxnet
> * Numpy
> * Opencv

------

### 操作步骤
#### 1.生成车牌sample[^code]
```
python genPlate.py 100 /Users/shelter/plate_100

参数1：生成车牌的数量
参数2：生成车牌存放的地址
```

#### 2.训练CNN模型[^code]
```
python train.py 
```

#### 3.预测车牌准确率
```
#随机生成100张车牌图片
python genPlate.py 100 /Users/shelter/test

#批量预测测试图片准确率
python test.py /Users/shelter/test

##输出结果示例
output:
预测车牌号码为:津 K 4 2 R M Y
输入图片数量:100
输入图片行准确率:0.72
输入图片列准确率:0.86
```

------
### 参考资料
```
1.http://blog.csdn.net/relocy/article/details/52174198
2.https://github.com/szad670401/learning-dl/tree/master/mxnet/ocr

```
