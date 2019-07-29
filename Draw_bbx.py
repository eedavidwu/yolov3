import cv2
import os
from os import listdir, getcwd
from os.path import join
from PIL import Image

wd = getcwd()

def draw_bbox(img_file, bbox=None, out_pic=None):
  img = Image.open(img_file)
  img.crop((bbox[0], bbox[1], bbox[2], bbox[3])).save(out_pic, "JPEG", quality=100)
  print('save into',out_pic)
 

def draw_all_bboxes(result,out_image_folder=None,out_label_folder=None):
  with open(result, 'r') as f:
    results_list = f.readlines()
  count=0
  for line in results_list:
    line=line.split()
    if (float(line[1])>0.985):
      count=count+1
      image_file=os.path.join(wd,'images',line[0]+'.jpg')
      bbox=[int(float(line[2])), int(float(line[3])), int(float(line[4])), int(float(line[5]))]
      
      #print (line)
      #print(bbox)
      out_pic= os.path.join(out_image_folder,line[0]+'_num'+str(count)+'.jpg')
      out_label=os.path.join(out_label_folder,line[0]+'_num'+str(count)+'.txt')
      draw_bbox(image_file, bbox, out_pic)
      label='0 0'
      label_out_file = open(out_label, 'w')
      label_out_file.write(label)
      print('save into',out_label)
      label_out_file.close()




if __name__ == "__main__":
  result = "./results/results.txt"
  out_image_folder = "./worker_image/"
  out_label_folder='./worker_labels/'
  draw_all_bboxes(result,out_image_folder,out_label_folder)

