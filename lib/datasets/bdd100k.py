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
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import json

class bdd100k(imdb):
  def __init__(self, image_set, use_diff=False):
    name = 'bdd100k' + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)

    self._image_set = image_set
    if 'train' in self._image_set:
      self.mode = 'train'
    elif 'val' in self._image_set:
      self.mode = 'val'
    
    self._devkit_path = self._get_default_path()
    self._data_path = os.path.join(self._devkit_path, 'images', '100k')
    imset_folder = self._image_set.replace('train', '').replace('val', '')

    self._classes = ('__background__',  # always index 0
                     'bike',
                     'bus',
                     'car',
                     'motor',
                     'person',
                     'rider',
                     'traffic light',
                     'traffic sign',
                     'train',
                     'truck')
    
    print('Num Classes:', len(self._classes))
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = ''
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

    if 'synth' in self._image_set.lower():
      self._image_set = self._image_set.replace('train', '')
      self._image_set = self._image_set.replace('val', '')
      image_path = os.path.join(self._devkit_path, self._image_set+'_images',
                              index + self._image_ext)
    else:
      image_path = os.path.join(self._data_path, 
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

    image_set_file = os.path.join(self._devkit_path, 'labels', 'ImageSets',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    
    return image_index

  def _get_default_path(self):
    """
    Return the default path where KITTI is expected to be installed.
    """
    #return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
    return os.path.join(cfg.DATA_DIR, 'bdd100k')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_ann = {}
    with open(os.path.join(self._devkit_path, 'labels', 'bdd100k_labels_images_%s.json'%self.mode), 'r') as f:
      annots = json.load(f)
      for ann in annots:
        gt_ann[self.mode+'/'+ann['name']] = ann['labels']

    gt_roidb = [self._load_bdd100k_annotation(gt_ann[index])
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

  def _load_bdd100k_annotation(self, labels):
    """
    Load image and bounding boxes info from bdd100k
    """
    num_objs = len(labels)
    gt_classes = np.zeros(num_objs, dtype=np.int32)
    gt_bboxes = np.zeros((num_objs,4), dtype=np.float32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for KITTI is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    ix = 0

    for ll in labels:
      clsName = ll['category']

      if clsName in ['drivable area', 'lane']:
        continue
      elif clsName not in self._classes:
        gt_classes = gt_classes[:-1]
        gt_bboxes = gt_bboxes[:-1]
        overlaps = overlaps[:-1]
        seg_areas = seg_areas[:-1]
        continue
      x1 = ll['box2d']['x1']
      x2 = ll['box2d']['x2']
      y1 = ll['box2d']['y1']
      y2 = ll['box2d']['y2']
      assert x1 >= 0 and x2 >=0 and y1 >= 0 and y2 >= 0
      assert x1 <= x2 and y1 <= y2
      clss = self._class_to_ind[clsName]
      gt_classes[ix] = clss
      gt_bboxes[ix,:] = [x1, y1, x2, y2]
      overlaps[ix, clss] = 1.0
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
      self._devkit_path,
      'results',
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} bdd100k results file'.format(cls))
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
      'labels',
      'bdd100k_labels_images_%s.json'%self.mode)
    imagesetfile = os.path.join(
      self._devkit_path,
      'labels',
      'ImageSets',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    rc = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False#True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap, rec_ALL = voc_eval(
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
