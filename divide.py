import os
from os import listdir, getcwd
from os.path import join
import pickle
from PIL import Image

train_percent=0.8
test_percent=0.1
val_percent=0.1

Img_folder = '/home/wuhaotian/tiny_yolo/darknet/person_obj/images'
train_dest = '/home/wuhaotian/tiny_yolo/darknet/person_obj/ImageSets/train.txt'
val_dest = '/home/wuhaotian/tiny_yolo/darknet/person_obj/ImageSets/val.txt'
test_dest = '/home/wuhaotian/tiny_yolo/darknet/person_obj/ImageSets/test.txt'

file_list = os.listdir(Img_folder)
num=len(file_list)
print (num)

train_number=(int(num*train_percent))
test_number=int(num*test_percent)
val_number=(num-train_number-test_number)

#out_file_all = open('all_label.txt', 'w').close()

train_list = open(train_dest, 'w')
val_list = open(val_dest, 'w')
test_list = open(test_dest, 'w')

i=0
for file_obj in file_list:
    file_path = os.path.join(Img_folder, file_obj)
    file_name, file_extend = os.path.splitext(file_obj)

    if (i< (train_number-1)):
        train_list.write(file_name + '\n')
    elif ((i< train_number + test_number-1)):
        val_list.write(file_name + '\n')
    elif((i< num)):
        test_list.write(file_name + '\n')

    i=i+1
train_list.close()
val_list.close()
test_list.close()

##after dividing train and test then convert the anno:

sets=[('train'), ('val'), ('test')]

classes = ["person"]
wd = getcwd()

def delete_useless(value,image_id):
    if len(value)==0:
        print ('Anno/%s.txt'%(image_id))
        print value
        os.remove('Anno/%s.txt'%( image_id))
        os.remove('images/%s.jpg'%( image_id))
    return 0


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('anno/%s.txt'%( image_id))
    out_file = open('labels/%s.txt'%(image_id), 'w')
    value=in_file.readline().split()
    #delete_useless(value,image_id)

    img=Image.open('images/%s.jpg'%(image_id))
    w=img.size[0]
    h=img.size[1]
    #b is defined as [x_min,x_max,y_min,y_max]
    b = (float(value[1]), float(value[3]), float(value[2]), float(value[4]))
    bb = convert((w, h), b)
    cls=value[0]
    cls_id = classes.index(cls)
    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file_all.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

out_file_all= open('all_label.txt', 'w').close
for image_set in sets:
    image_ids = open('/home/wuhaotian/darknet/person_data/ImageSets/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    out_file_all= open('all_label.txt', 'a')
    for image_id in image_ids:
        list_file.write('%s/images/%s.jpg\n'%(wd, image_id))
        convert_annotation(image_id)


os.system("cat train.txt val.txt >  train_final.txt")

