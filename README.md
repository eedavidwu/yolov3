YOLOV3参数理解：
https://blog.csdn.net/weixin_42731241/article/details/81474920



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
### For cfg:
<br>参数理解：https://blog.csdn.net/weixin_42731241/article/details/81474920
<br> for the training: batch=64 subdivisions=8
<br> (bofore yolo) filters= num*(class+5)/3 (3处)
<br> 修改anchors,classes (3处)

### For person_obj.names:
<br>person
<br>cat
<br>等等

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


