reference:https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
# 下载与测试darknet yolo：
<br>git clone https://github.com/pjreddie/darknet
<br>cd darknet
<br>修改Makefiles文件，CUDA,OPENCV,GPU
<br>make

## 测试下载的YOLO内容(下载权重后下载测试图片检测)：
<br> for v3:
<br> wget https://pjreddie.com/media/files/yolov3.weights
<br> ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.5

<br> for tiny:
<br> wget https://pjreddie.com/media/files/yolov3-tiny.weights
<br> ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg -thresh 0.5


## For webcam and video:
<br> for the webcam:
<br> ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
<br> for the video:
<br>./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>

# 训练自己的数据集
## 制作数据集、统一数据格式
<br> 首先制作voc格式数据集，本质上就是1.jpg-1.txt, 内容：0 0.1 0.2 0.1 0.2 归一化的(x,y,w,h) 使用voc_label.py or divide.py 
<br>（先分割数据集，然后把分割后不带后缀的文件名写在train.txt内，将jpg文件路径写在train.txt/val.txt里，最后合并为train_final，
在之后我们把对应的数据集txt转化成VOC格式,并将对应的txt，放在labels/%.txt里，最后合并到all_labels.txt）

## 训练数据预处理
<br>使用gen_anchor.py,生成k-means得到的anchor,
<br>需要修改内容：width&height: 32倍数和cfg一致，默认416。
<br>修改argument, filelist（all_label.txt地址，包含所有的label内容），
<br>修改output_dir (输出文件地址)，
<br>修改num_clusters：聚类类数量

## 修改cfg,data,name文件
### For yolo-obj.cfg:
<br> for the training: batch=64 subdivisions=8
<br> (bofore yolo) filters= num*(class+5)/3 (3处)
<br> 修改anchors,classes (3处)
<br>参数理解：https://blog.csdn.net/weixin_42731241/article/details/81474920
<br>https://blog.csdn.net/jinlong_xu/article/details/76375334,
<br>if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power); 

####（具体的内容：）
<br> cp yolov3.cfg to yolo-obj.cfg 并修改 batch=64, subdivisions=8
<br> 修改max_batches to (classes*2000), f.e. max_batches=6000 if you train for 3 classes
<br> change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
<br> change line classes=1 to your number of objects in each of 3 [yolo] layers:
<br> change [filters=255] to filters=(classes + 5)xnum/3 in the 3 [convolutional] before each [yolo] layer. So if classes=1 then should be filters=18. If classes=2 then write filters=21. (Generally filters depends on the classes, coords and number of masks, i.e. filters=(classes + coords + 1)*<number of mask>, where mask is indices of anchors. If mask is absence, then filters=(classes + coords + 1)*num)

### For person_obj.names:
<br> Create obj.names in the directory with objects names - each in new line 
<br> 如：
<br>person
<br>cat



### For person_obj.data::
<br>classes= 1
<br>train  = /home/wuhaotian/tiny_yolo/darknet/person_obj/train.txt
<br>valid  = /home/wuhaotian/tiny_yolo/darknet/person_obj/val.txt
<br>names  = /home/wuhaotian/tiny_yolo/darknet/person_obj/cfg/person_obj.names
<br>backup = /home/wuhaotian/tiny_yolo/darknet/backup

## 下载预训练权重：
<br> wget https://pjreddie.com/media/files/darknet53.conv.74\

## 开始训练：
<br> ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74


