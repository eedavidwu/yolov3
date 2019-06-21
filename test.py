#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ssh.py    
@Contact :   hupeng25@huawei.com
@License :   (C)Copyright 2018-2019

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2/18/19 7:08 AM   gxrao      1.0         None
'''
# test code:
echo "pedestrian_detect"
python yolo_eval.py \
    --detpath=./data/pedestrian_train/pedestrian_detect_tiny_yolov3person.txt \
    --imagesetfile=/datasets/pedestrian_dataset/yolo_style/shuffle_test_in.txt\
    --classname='0'
#code
import numpy as np
import os
import cPickle
from argparse import ArgumentParser

def get_file_name(in_path):
    label_name = os.path.basename(in_path).split('.jpg')[0]
    return label_name

def get_file_name_suffix(in_path):
    label_name = os.path.basename(in_path)
    return label_name

def parser():
    parser = ArgumentParser("Map calculator")
    parser.add_argument("--detpath", dest="in_detpath", help='Path to detections',
                        default='./data/detect_new_poolperson.txt', type=str)
    parser.add_argument("--imagesetfile", dest="in_imageset", help='text file containing the list of images, one image per line',
                        default='/datasets/shoulder_data/test_ordinary/test_in.txt', type=str)
    parser.add_argument("--classname", dest="in_classname", help='Category name (duh)',
                        default='person', type=str)
    parser.add_argument("--cachedir", dest="in_cachedir", help='Directory for caching the annotations',
                        default='./data', type=str)
    parser.add_argument("--ovthresh", dest="th", help='Overlap threshold (default = 0.5)',
                        default='0.5', type=float)
    return parser.parse_args()

def parse_rec(filepath):
    """
    Parse a ssh xml file
    style:
        person x1 y1 x2 y2
        person x1 y1 x2 y2
    """
    object = []
    with open(filepath) as fd:
        lines = fd.readlines()
        bbox = [line.strip().split(' ')[1:] for line in lines]
        class_name = [line.strip().split(' ')[0] for line in lines]
    for i, bbox in enumerate(bbox):
        obj_struct = {}
        obj_struct['name'] = class_name[i]
        obj_struct['bbox'] = list(map(float, bbox))
        object.append(obj_struct)
    return object

def parse_rec(filepath, imgpath):
    """
        Parse a yolo xml file
        style:
            person x1 y1 x2 y2
            person x1 y1 x2 y2
        """
    object = []
    import cv2
    img = cv2.imread(imgpath)
    im_height, im_width = img.shape[0], img.shape[1]
    with open(filepath) as fd:
        lines = fd.readlines()
        ssh_bbox = [line.strip().split(' ')[1:] for line in lines]
        bbox = []
        for line in ssh_bbox:
            line = [float(val) for val in line]
            x_l = int((line[0] - line[2] / 2.0) * im_width + 1)
            x_r = int((line[0] + line[2] / 2.0) * im_width + 1)
            y_l = int((line[1] - line[3] / 2.0) * im_height + 1)
            y_h = int((line[1] + line[3] / 2.0) * im_height + 1)
            bbox.append([x_l, y_l, x_r, y_h])
        class_name = [line.strip().split(' ')[0] for line in lines]
    for i, bbox in enumerate(bbox):
        obj_struct = {}
        obj_struct['name'] = class_name[i]
        obj_struct['bbox'] = list(map(float, bbox))
        object.append(obj_struct)
    return object

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    usage:
        heardshoulder detection ap calculation
        '''
        darknet :AP for person = 0.8805  with_longgang= 0.8756
        '''
        save_txt = r'/home/hupeng/data/ordinary/ssh_test_in.txt'
        detpath = r'./data/detect_result.txt'
        annopath = r'/home/hupeng/data/ordinary/annotations/{}.txt'
        classname = 'person'
        cachedir = './data'
        voc_eval(detpath,
                 annopath,
                 save_txt,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=False)
        save_txt can get from the from following code
        set_path = r'/home/hupeng/data/with_test/image_set/test_in.txt'
        image_path = r'/home/hupeng/data/ordinary/image'
        save_txt = r'/home/hupeng/data/ordinary/ssh_test_in.txt'
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    if os.path.exists(cachefile):
        os.remove(cachefile)
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(imagename.replace('.jpg', '.txt').replace('images', 'annotations').replace('JPEGImages','labels'),
                                        imagename)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        num_np = np.array([True for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(num_np)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print('AP for {} = {:.4f}'.format('person', ap))
    return rec, prec, ap

if __name__ == "__main__":
    '''
    darknet :AP for person = 0.8805
    '''
    args = parser()

    detpath = args.in_detpath
    save_txt = args.in_imageset
    classname = args.in_classname
    cachedir = args.in_cachedir
    th = args.th
    voc_eval(detpath,
             save_txt,
             classname,
             cachedir,
             ovthresh=th,
             use_07_metric=False)
