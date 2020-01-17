# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .cityscapes_eval import cityscapes_eval
from model.config import cfg
import json

class cityscapes(imdb):
  def __init__(self, image_set, use_diff=False):
    name = 'cityscapes' + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    self._data_path = os.path.join(self._devkit_path)
    if 'synthFoggy' in image_set:
      with open(os.path.join(cfg.ROOT_DIR, "trained_weights/netD_CsynthFoggyC_score.json"), "r") as read_file:
        self.D_T_score = json.load(read_file)
    if 'synthBDD' in image_set:
      with open(os.path.join(cfg.ROOT_DIR, "trained_weights/netD_CsynthBDDday_score.json"), "r") as read_file:
        self.D_T_score = json.load(read_file)
    if cfg.ADAPT_MODE == 'K2C':
      self._classes = ('__background__', 'car')# always index 0
    elif cfg.ADAPT_MODE == 'C2F':
      self._classes = ('__background__',  # always index 0
                     'person',
                     'rider',
                     'car',
                     'truck',
                     'bus',
                     'train',
                     'motorcycle',
                     'bicycle'
                      )
    elif cfg.ADAPT_MODE == 'C2BDD':
      # classes correspond to bdd100k
      self._classes = ('__background__',  # always index 0
                      'bicycle',
                      'bus',
                      'car',
                      'motorcycle',
                      'person',
                      'rider',
                      'traffic light',
                      'traffic sign',
                      'train',
                      'truck') 
    print(self._classes) 
    print('Num Classes:', len(self._classes))
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    loc = index[:index.find('_')]    
    image_path = os.path.join(self._data_path, 'leftImg8bit', self._image_set, loc,
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'leftImg8bit', 
                                  self._image_set)
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    image_index = []
    for folder in os.listdir(image_set_file):
      imgs = os.listdir(image_set_file + '/' + folder)
      imgs = [img[:img.find('leftImg8bit')+11] for img in imgs]
      if 'foggy' in self._image_set:
        imgs = sorted(imgs)
        for i in range(len(imgs)):
          if (i % 3) == 0:
            imgs[i] += '_foggy_beta_0.005'
          elif (i % 3) == 1:
            imgs[i] += '_foggy_beta_0.01'
          elif (i % 3) == 2:
            imgs[i] += '_foggy_beta_0.02'
      image_index.extend(imgs)
    # with open(image_set_file) as f:
    #   image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where cityscapes is expected to be installed.
    """
    #return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
    return os.path.join(cfg.DATA_DIR, 'CityScapes')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '%dclasses'%len(self._classes) +  '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name + '%dclasses'%len(self._classes), cache_file))
      return roidb

    gt_roidb = [self._load_cityscapes_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_cityscapes_annotation(self, index):
    """
    Load image and bounding boxes info from cityscapes
    """
    loc = index[:index.find('_')]
    index = index[:index.find('leftImg8bit')]
    filename = os.path.join(self._data_path, 'gtFine', self._image_set, loc, index + 'gtFine_polygons.json')
    with open(filename, 'r') as f:
        info = json.load(f)
    objs = info["objects"]
    num_objs = len(objs)
    gt_classes = np.zeros(num_objs, dtype=np.int32)
    #gt_trunc = np.zeros(num_objs, dtype=np.float32)
    gt_bboxes = np.zeros((num_objs,4), dtype=np.float32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for cityscapes is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    #print num_objs
    ix = 0
    for obj in objs:
        clsName = obj['label']
        #print clsName
        if clsName.lower().strip() not in self._classes:
            gt_classes = gt_classes[:-1]
            #gt_trunc = gt_trunc[:-1]
            #gt_occlude = gt_occlude[:-1]
            #gt_alpha = gt_alpha[:-1]
            gt_bboxes = gt_bboxes[:-1]
            overlaps = overlaps[:-1]
            seg_areas = seg_areas[:-1]
            continue
        
        maxW = float(info['imgWidth']) - 1.
        maxH = float(info['imgHeight']) - 1.
        x1 = maxW
        y1 = maxH
        x2 = 0.
        y2 = 0.
        for p in obj['polygon']: # (x, y)
          if p[0] < x1:
            x1 = max(0, p[0])
          if p[0] > x2:
            x2 = min(maxW, p[0])
          if p[1] < y1:
            y1 = max(0, p[1])
          if p[1] > y2:
            y2 = min(maxH, p[1])
        assert x1 >= 0 and x2 >=0 and y1 >= 0 and y2 >= 0
        assert x1 <= x2 and y1 <= y2
        cls = self._class_to_ind[clsName.lower().strip()]
        gt_classes[ix] = cls
        #gt_trunc[ix] = trunc
        #gt_occlude[ix] = occlude
        #gt_occ[ix] = occ
        #gt_alpha[ix] = alpha
        gt_bboxes[ix,:] = [x1, y1, x2, y2]
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        ix+=1

    overlaps = scipy.sparse.csr_matrix(overlaps)


    return {'boxes': gt_bboxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      #self._devkit_path,
      './',
      'results',
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} cityscapes results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'gtFine',
      self._image_set,
      '{:s}',
      '{:s}gtFine_polygons.json')
    imagesetfile = self._image_index
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False#True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      ##rec_ALL is counting every predicted tp boxes (including overlaps)
      rec, prec, ap = cityscapes_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join('/home/disk1/DA/pytorch-faster-rcnn', 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True
