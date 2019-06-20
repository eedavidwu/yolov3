from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import sys
import os
import shutil
import random 
import math
from PIL import Image

width_in_cfg_file = 416.  # input width of network model
height_in_cfg_file = 416. # input height of network model

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:
            # print(w)
            # print(h)
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(IOU(X[i],centroids)) 
    return sum / n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')  
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file
        anchors[i][1] *= height_in_cfg_file
    
    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)
    print('Anchors = ', anchors[sorted_indices]) 
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    f.write('%f\n'%(avg_IOU(X,centroids)))

def kmeans(X,centroids,eps,anchor_file):
    
    N = X.shape[0]
    iterations = 0
    k = centroids.shape[0]
    prev_assignments = np.ones(N) * (-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        print(assignments)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)            
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,2),np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j] / (np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default = '/home/wuhaotian/tiny_yolo/darknet/person_obj/all_label.txt',
                        help='path to filelist\n')
    parser.add_argument('-output_dir', default = '/home/wuhaotian/tiny_yolo/darknet/person_obj/', type = str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default = 9, type = int, help='number of clusters\n')

    args = parser.parse_args()    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    annotation_dims = []
    with open(args.filelist) as fp:
        for line in fp:
            corData = line.strip().split(' ') # line: image_path class xcenter ycenter wnorm hnorm
            # xc = float(corData[1])
            # yc = float(corData[2])
            # w = float(corData[3])
            # h = float(corData[4])

            width, height = 416, 416
            # w = (x2 - x1) / width
            # h = (y2 - y1) / height
            w = float(corData[3])
            h = float(corData[4])
            if(w * width > 64 or h*height > 64): # this condition is added to avoid small objects are taking into account, modified it if you like
                annotation_dims.append(list(map(float,(w,h))))
    annotation_dims = np.array(annotation_dims)
    print(annotation_dims.shape)
    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( args.output_dir,'anchors%d.txt'%(num_clusters))
            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        print('centroids.shape', centroids.shape)
        kmeans(annotation_dims,centroids,eps,anchor_file)

if __name__=="__main__":
    main(sys.argv)
