
"""Data loader and processing."""
import torch
import torch.nn.functional as f

import random
import os
import numpy as np
import glob
import random
import cv2

import json
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pycocotools import mask
import glob
import copy
import mmcv

def segmentationToCocoMask(labelMap, labelId):
  '''
  Encodes a segmentation mask using the Mask API.
  :param labelMap: [h x w] segmentation map that indicates the label of each pixel
  :param labelId: the label from labelMap that will be encoded
  :return: Rs - the encoded label mask for label 'labelId'
  '''
  labelMask = labelMap == labelId
  labelMask = np.expand_dims(labelMask, axis=2)
  labelMask = labelMask.astype('uint8')
  labelMask = np.asfortranarray(labelMask)
  Rs = mask.encode(labelMask)
  assert len(Rs) == 1
  Rs = Rs[0]

  return Rs

def semaug_proc(coco, coco_stuff, image_id, idx, anns, I, track, embeddings_index, d, obj_class_names, embedding_dim): 
  words = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'stoplight',
           'hydrant', 'street', 'stop', 'parking', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag',
           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'ball', 'kite', 'baseball', 'skateboard',
           'surfboard', 'racket', 'bottle', 'plate', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
           'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'bed', 'mirror', 'desk', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
           'teddybears', 'hairdryer', 'toothbrush', 'hairbrush', 'banner', 'blanket', 'branch', 'bridge', 'building',
           'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'cloth', 'clothes', 'clouds',
           'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'stuff', 'door', 'fence', 'floor', 'marble', 'tile',
           'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground', 'hill', 'house', 'leaves',
           'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement',
           'pillow', 'plant', 'plastic', 'platform', 'field', 'railing', 'railroad', 'river', 'road', 'rock',
           'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone',
           'straw', 'structural', 'tent', 'textile', 'other', 'towel', 'tree', 'vegetable', 'wall', 'brick',
           'concrete', 'panel', 'wood', 'water', 'droplets', 'window', 'blind', 'plane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person', 'plant',
           'sheep', 'sofa', 'train', 'tv']


  annIds_stuff = coco_stuff.getAnnIds(imgIds=[image_id])
  anns_stuff = coco_stuff.loadAnns(annIds_stuff)

  # If there are no segmentations for this image then skip augmentation
  if not anns and not anns_stuff: return I, anns

  # Create a list of the segmentations
  seglist = []
  seglist_obj = []

  for i in range(len(anns)):
    seglist_obj.append(d[anns[i]['category_id']])
    seglist.append(d[anns[i]['category_id']])
  for i in range(len(anns_stuff)):
    if anns_stuff[i]['category_id'] < 183:
      seglist.append(d[anns_stuff[i]['category_id']])

  # Remove duplicates
  seglist = list(dict.fromkeys(seglist))
  seglist_obj = list(dict.fromkeys(seglist_obj))

  # Prepare embedding matrix
  hits = 0
  misses = 0
  inputs = str(seglist).strip('[').strip(']').replace("'", "")
  inputs2 = inputs.replace("-", ", ")
  input_words = inputs2.split(", ")
  input_words2 = inputs.split(", ")
  input_obj = str(seglist_obj).strip('[').strip(']').replace("'", "")
  input_obj2 = input_obj.replace("-", ", ")
  input_obj3 = input_obj2.split(", ")
  while 'other' in input_words:
    input_words.remove('other')
  while 'other' in input_obj3:
    input_obj3.remove('other')
  embedding_matrix_sample = np.zeros((len(input_words), embedding_dim))
  embedding_matrix_sample_stuff = np.zeros((len(input_words), embedding_dim))
  
  # Some words in coco don't exist in the GloVe embeddings, so find the closest
  for i, word in enumerate(seglist):
    stuffnum = [k for (k, v) in d.items() if v == word]
    if word == 'potted plant':
      word = 'plant'
    elif word == 'dining table':
      word = 'table'
    elif word == 'eye glasses':
      word = 'eyeglasses'
    elif word == 'playingfield':
      word = 'field'
    elif word == 'waterdrops':
      word = 'droplets'
    elif word == 'cell phone':
      word = 'cellphone'
    elif word == 'teddy bear':
      word = 'teddybears'
    elif word == 'wine glass':
      word = 'wineglass'
    elif word == 'traffic light':
      word = 'stoplight'
    elif word == 'fire hydrant':
      word = 'hydrant'
    elif word == 'sports ball':
      word = 'ball'
    elif word == 'tennis racket':
      word = 'racket'
    elif word == 'stop sign':
      word = 'stoplight'
    elif word == 'parking meter':
      word = 'parking'
    elif word == 'baseball bat':
      word = 'baseball'
    elif word == 'baseball glove':
      word = 'baseball'
    elif word == 'hot dog':
      word = 'hotdog'
    elif word == 'hair brush':
      word = 'hairbrush'
    elif word == 'hair drier':
      word = 'hairdryer'
    elif word == 'door-stuff':
      word = 'door'
    elif word == 'desk-stuff':
      word = 'desk'
    elif word == 'food-other':
      word = 'food'
    elif word == 'plant-other':
      word = 'plant'
    elif word == 'sky-other':
      word = 'sky'
    elif word == 'structural-other':
      word = 'structural'
    elif word == 'textile-other':
      word = 'textile'
    elif word == 'ground-other':
      word = 'ground'
    elif word == 'mirror-stuff':
      word = 'mirror'
    elif word == 'furniture-other':
      word = 'furniture'
    elif word == 'floor-wood' or word == 'floor-marble' or word == 'floor-other' or word == 'floor-stone' or word == 'floor-tile':
      word = 'floor'
    elif word == 'wall-wood' or word == 'wall-brick' or word == 'wall-other' or word == 'wall-stone' or word == 'wall-tile' or word == 'wall-concrete' or word == 'wall-panel':
      word = 'wall'
    elif word == 'window-blind' or word == 'window-other':
      word = 'window'
    elif word == 'solid-other':
      word = 'solid'
    elif word == 'building-other':
      word = 'building'
    elif word == 'ceiling-other' or word == 'ceiling-tile':
      word = 'ceiling'
    elif word == 'other':
      continue
    elif word == 'background':
      continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      if stuffnum:
        if int(stuffnum[0]) > 90:
          embedding_matrix_sample_stuff[i] = embedding_vector
      hits += 1
    else:
      misses += 1


  # Some words in coco don't exist in the GloVe embeddings, so find the closest
  for i, word in enumerate(input_words):
    if word == 'potted plant':
      word = 'plant'
    elif word == 'dining table':
      word = 'table'
    elif word == 'eye glasses':
      word = 'eyeglasses'
    elif word == 'playingfield':
      word = 'field'
    elif word == 'waterdrops':
      word = 'droplets'
    elif word == 'cell phone':
      word = 'cellphone'
    elif word == 'teddy bear':
      word = 'teddybears'
    elif word == 'wine glass':
      word = 'wineglass'
    elif word == 'traffic light':
      word = 'stoplight'
    elif word == 'fire hydrant':
      word = 'hydrant'
    elif word == 'sports ball':
      word = 'ball'
    elif word == 'tennis racket':
      word = 'racket'
    elif word == 'stop sign':
      word = 'stoplight'
    elif word == 'parking meter':
      word = 'parking'
    elif word == 'baseball bat':
      word = 'baseball'
    elif word == 'baseball glove':
      word = 'baseball'
    elif word == 'hot dog':
      word = 'hotdog'
    elif word == 'hair brush':
      word = 'hairbrush'
    elif word == 'hair drier':
      word = 'hairdryer'
    elif word == 'other':
      continue
    elif word == 'background':
      continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix_sample[i] = embedding_vector
      hits += 1
    else:
      misses += 1
      print("WORD IS MISSING", word)

  embedding_matrix_objects = np.ones((len(obj_class_names), embedding_dim)) * 1000
  embedding_matrix_objects_cos = np.zeros((len(obj_class_names), embedding_dim))

  # Find the embedding vector for the object to be pasted
  for idx_obj, word in enumerate(obj_class_names):
    if word not in input_words:
      if word == 'potted plant':
        word = 'plant'
      elif word == 'dining table':
        word = 'table'
      elif word == 'aeroplane':
        word = 'plane'
      elif word == 'diningtable':
        word = 'table'
      elif word == 'pottedplant':
        word = 'plant'
      elif word == 'tvmonitor':
        word = 'tv'
      elif word == 'playingfield':
        word = 'field'
      elif word == 'eye glasses':
        word = 'eyeglasses'
      elif word == 'cell phone':
        word = 'cellphone'
      elif word == 'waterdrops':
        word = 'droplets'
      elif word == 'teddy bear':
        word = 'teddybears'
      elif word == 'wine glass':
        word = 'wineglass'
      elif word == 'traffic light':
        word = 'stoplight'
      elif word == 'fire hydrant':
        word = 'hydrant'
      elif word == 'sports ball':
        word = 'ball'
      elif word == 'baseball bat':
        word = 'baseball'
      elif word == 'baseball glove':
        word = 'baseball'
      elif word == 'tennis racket':
        word = 'racket'
      elif word == 'stop sign':
        word = 'stoplight'
      elif word == 'hair brush':
        word = 'hairbrush'
      elif word == 'parking meter':
        word = 'parking'
      elif word == 'hair drier':
        word = 'hairdryer'
      elif word == 'hot dog':
        word = 'hotdog'
      elif word == 'background':
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix_objects[idx_obj] = embedding_vector
        embedding_matrix_objects_cos[idx_obj] = embedding_vector
        hits += 1
      else:
        misses += 1
        print("WORD IS MISSING:", word)


  embedding_matrix_objects_cos = torch.from_numpy(embedding_matrix_objects_cos)
  embedding_matrix_sample = torch.from_numpy(embedding_matrix_sample)
  normed_embedding = f.normalize(embedding_matrix_objects_cos,dim=1,p=2)
  normed_array = f.normalize(embedding_matrix_sample,dim=1,p=2) 

  cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
  closest_cosine_words = torch.topk(cosine_similarity, k=3)
  closest_cosine_words = closest_cosine_words.indices[0].numpy()
  potential_objects = []

  for word_idx in range(0, 3):
    whatword_cos = obj_class_names[closest_cosine_words[word_idx]]

    # Get the ID number of the location
    if whatword_cos == 'diningtable':  # to ensure it isn't placed on vegetable instead of dining table or table-other
      wherenum = [11]
    elif whatword_cos == 'pottedplant':
      wherenum = [16]
    elif whatword_cos == 'tvmonitor':
      wherenum = [20]
    elif whatword_cos == 'aeroplane':
      wherenum = [1]
    else:
      wherenum = [idx for idx, key in enumerate(list(d.items())) if
                  whatword_cos in key[1]]
    if wherenum[0] > 0:
      potential_objects.append(wherenum[0])
    else:
      print("whatword_cos:", whatword_cos)


  objectList = []
  for i in range(0, len(potential_objects)):
    objectList.append(track[potential_objects[i]])
  objIdx = objectList.index(np.min(objectList))
  track[potential_objects[objIdx]] += 1

  # Find the embedding vector for the object to be pasted
  word = obj_class_names[closest_cosine_words[objIdx]] 
  if word == 'potted plant':
    word = 'plant'
  elif word == 'dining table':
    word = 'table'
  elif word == 'aeroplane':
    word = 'plane'
  elif word == 'diningtable':
    word = 'table'
  elif word == 'pottedplant':
    word = 'plant'
  elif word == 'tvmonitor':
    word = 'tv'
  elif word == 'playingfield':
    word = 'field'
  elif word == 'eye glasses':
    word = 'eyeglasses'
  elif word == 'cell phone':
    word = 'cellphone'
  elif word == 'waterdrops':
    word = 'droplets'
  elif word == 'teddy bear':
    word = 'teddybears'
  elif word == 'wine glass':
    word = 'wineglass'
  elif word == 'traffic light':
    word = 'stoplight'
  elif word == 'fire hydrant':
    word = 'hydrant'
  elif word == 'sports ball':
    word = 'ball'
  elif word == 'baseball bat':
    word = 'baseball'
  elif word == 'baseball glove':
    word = 'baseball'
  elif word == 'tennis racket':
    word = 'racket'
  elif word == 'stop sign':
    word = 'stoplight'
  elif word == 'hair brush':
    word = 'hairbrush'
  elif word == 'parking meter':
    word = 'parking'
  elif word == 'hair drier':
    word = 'hairdryer'
  elif word == 'hot dog':
    word = 'hotdog'

  embedding_vector = embeddings_index.get(word).astype(dtype="float64")

  # Find the location of object placement through cosine similarity
  embedding_matrix_sample_stuff = torch.from_numpy(embedding_matrix_sample_stuff)
  embedding_vector = torch.from_numpy(embedding_vector)
  normed_embedding = f.normalize(embedding_matrix_sample_stuff,dim=-1,p=2) 
  normed_array = f.normalize(embedding_vector.reshape((1, len(embedding_vector))),dim=-1,p=2) 

  cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
  closest_k_words = torch.topk(cosine_similarity, k=1)
  closest_k_words = closest_k_words.indices[0].numpy()


  whereword = [k for (k, v) in embeddings_index.items() if
               np.allclose(v, embedding_matrix_sample_stuff[closest_k_words[0]])]

  if whereword:
    # Get the ID number of the location
    if whereword[0] == 'table':  # to ensure it isn't placed on vegetable instead of dining table or table-other
      wherenum = [67, 165]
    elif whereword[0] == 'cup':  # to ensure it isn't placed on cupboard instead of cup
      wherenum = [47]
    elif whereword[0] == 'cellphone':  # to ensure it isn't placed on cupboard instead of cup
      wherenum = [77]
    elif whereword[0] == 'plant':
      wherenum = [64]
    elif whereword[0] == 'table':
      wherenum = [67, 165]
    elif whereword[0] == 'field':
      wherenum = [145]
    elif whereword[0] == 'droplets':
      wherenum = [179]
    elif whereword[0] == 'teddybears':
      wherenum = [88]
    elif whereword[0] == 'wineglass':
      wherenum = [46]
    elif whereword[0] == 'eyeglasses':
      wherenum = [30]
    elif whereword[0] == 'stoplight':
      wherenum = [10]
    elif whereword[0] == 'hydrant':
      wherenum = [11]
    elif whereword[0] == 'ball':
      wherenum = [37]
    elif whereword[0] == 'racket':
      wherenum = [43]
    elif whereword[0] == 'stop':
      wherenum = [13]
    elif whereword[0] == 'parking':
      wherenum = [14]
    elif whereword[0] == 'hotdog':
      wherenum = [58]
    elif whereword[0] == 'baseball-reference':
      wherenum = [39, 40]
    elif whereword[0] == 'hairbrush':
      wherenum = [91]
    elif whereword[0] == 'hairdryer':
      wherenum = [89]
    elif whereword[0] == 'road':
      wherenum = [149]
    elif whereword[0] == 'door':
      wherenum = [71, 112]
    else:
      wherenum = [idx for idx, key in enumerate(list(d.items())) if
                  whereword[0] in key[1]]
  else:
    # Find the location of object placement through cosine similarity
    normed_embedding = f.normalize(embedding_matrix_sample, dim=1,
                                   p=2) 
    normed_array = f.normalize(embedding_vector.reshape((1, len(embedding_vector))), dim=1,
                               p=2) 

    cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
    closest_k_words = torch.topk(cosine_similarity, k=1)
    closest_k_words = closest_k_words.indices[0].numpy()

    whereword = [k for (k, v) in embeddings_index.items() if
                 np.allclose(v, embedding_matrix_sample[closest_k_words[0]])]

    # Get the ID number of the location
    if whereword[0] == 'table':  # to ensure it isn't placed on vegetable instead of dining table or table-other
      wherenum = [67, 165]
    elif whereword[0] == 'cup':  # to ensure it isn't placed on cupboard instead of cup
      wherenum = [47]
    elif whereword[0] == 'cellphone':  # to ensure it isn't placed on cupboard instead of cup
      wherenum = [77]
    elif whereword[0] == 'plant':
      wherenum = [64]
    elif whereword[0] == 'table':
      wherenum = [67, 165]
    elif whereword[0] == 'field':
      wherenum = [145]
    elif whereword[0] == 'droplets':
      wherenum = [179]
    elif whereword[0] == 'teddybears':
      wherenum = [88]
    elif whereword[0] == 'wineglass':
      wherenum = [46]
    elif whereword[0] == 'eyeglasses':
      wherenum = [30]
    elif whereword[0] == 'stoplight':
      wherenum = [10]
    elif whereword[0] == 'hydrant':
      wherenum = [11]
    elif whereword[0] == 'ball':
      wherenum = [37]
    elif whereword[0] == 'racket':
      wherenum = [43]
    elif whereword[0] == 'stop':
      wherenum = [13]
    elif whereword[0] == 'parking':
      wherenum = [14]
    elif whereword[0] == 'hotdog':
      wherenum = [58]
    elif whereword[0] == 'baseball-reference':
      wherenum = [39, 40]
    elif whereword[0] == 'hairbrush':
      wherenum = [91]
    elif whereword[0] == 'hairdryer':
      wherenum = [89]
    else:
      wherenum = [idx for idx, key in enumerate(list(d.items())) if
                  whereword[0] in key[1]]

  # Get the center coordinates for the object placement
  cent_x = 0
  cent_y = 0
  # If the location to paste is in the "stuff" category
  if int(wherenum[0]) > 91:
    for i in range(0, len(anns_stuff)):
      if anns_stuff[i]['category_id'] in wherenum:
        cent_x = anns_stuff[i]['bbox'][0] + (anns_stuff[i]['bbox'][2] / 2)
        cent_y = anns_stuff[i]['bbox'][1] + (anns_stuff[i]['bbox'][3] / 2)
    if cent_x == 0:
      for i in range(0, len(anns)):
        if anns[i]['category_id'] in wherenum:
          cent_x = anns[i]['bbox'][0] + (anns[i]['bbox'][2] / 2)
          cent_y = anns[i]['bbox'][1] + (anns[i]['bbox'][3] / 2)
  else:  # If the location to paste is in the "object" category
    for i in range(0, len(anns)):
      if anns[i]['category_id'] in wherenum:
        cent_x = anns[i]['bbox'][0] + (anns[i]['bbox'][2] / 2)
        cent_y = anns[i]['bbox'][1] + (anns[i]['bbox'][3] / 2)
    if cent_x == 0:
      for i in range(0, len(anns_stuff)):
        if anns_stuff[i]['category_id'] in wherenum:
          cent_x = anns_stuff[i]['bbox'][0] + (anns_stuff[i]['bbox'][2] / 2)
          cent_y = anns_stuff[i]['bbox'][1] + (anns_stuff[i]['bbox'][3] / 2)

  # Find the filenames of the objects in the bank and pick one randomly
  dirname = "data/cocoBank/"
  files = glob.glob(os.path.join(dirname, str(potential_objects[objIdx]), str(potential_objects[objIdx]), '*_ann.png'))

  area_ann = 0
  # Ensure the object to be placed is <300 pixels in area - keeps unrecognizable objects from being placed
  while area_ann < 300:
    try:
      rannum = random.randint(0, len(files) - 1)
      I_ann = mmcv.imread(files[rannum])
      I_annI = mmcv.imread(files[rannum].replace("_ann.png", ".png"))
      area_ann = np.sum(I_ann) / 255
    except:
      area_ann = 0

  # For Grayscale/RGB compatability issues
  if len(I_annI.shape) != len(I.shape):
    if len(I.shape) < len(I_annI.shape):
      I = np.stack((I, I, I), axis=2)
    else:
      I_annI = np.stack((I_annI, I_annI, I_annI), axis=2)

  # Create a mask image to check if pasted object covers any existing objects
  anns_img = np.zeros((I.shape[0], I.shape[1]))
  for ann in anns:
    anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])

  # Paste new object into image and into mask image
  [ann_y, ann_x, channels] = I_ann.shape
  index = 0
  indey = 0
  aug_area = 0
  for idx1 in range(int(cent_x - np.round((ann_x / 2))), int(cent_x + np.round((ann_x / 2))) - 1):
    for idy in range(int(cent_y - np.round((ann_y / 2))), int(cent_y + np.round((ann_y / 2))) - 1):
      if idx1 < I.shape[1] and idy < I.shape[0] and idx1 >= 0 and idy >= 0:
        if I_ann[indey, index, 0] == 255:
          I[idy, idx1] = I_annI[indey, index]
          anns_img[idy, idx1] = potential_objects[objIdx]
          aug_area += 1
      indey = indey + 1
    index = index + 1
    indey = 0

  Rs = segmentationToCocoMask(anns_img, potential_objects[objIdx])

  # Find new bounding box dimensions
  left_edge = np.maximum(0, int(cent_x - np.round((ann_x / 2))))
  right_edge = np.minimum(I.shape[1], int(cent_x + np.round((ann_x / 2))) - 1)
  top_edge = np.maximum(0, int(cent_y - np.round((ann_y / 2))))
  bottom_edge = np.minimum(I.shape[0], int(cent_y + np.round((ann_y / 2))) - 1)
  bbox_width = right_edge - left_edge
  bbox_height = bottom_edge - top_edge

  ann_idx = 0

  for ann in anns:
    [LE, TE, annWidth, annHeight] = ann['bbox']

    if (LE >= (left_edge + bbox_width)) or ((LE + annWidth) <= left_edge) or ((TE + annHeight) <= top_edge) or (
            TE >= (top_edge + bbox_height)):
      LE = 0  # rectangles do not overlap
      ann_idx += 1

    else:
      # MODIFY THE ANNOTATION AS NECESSARY
      anns_img2 = np.zeros((I.shape[0], I.shape[1]))
      anns_img2 = np.maximum(anns_img2, coco.annToMask(ann) * ann['category_id'])
      index = 0
      indey = 0
      for idx1 in range(int(cent_x - np.round((ann_x / 2))), int(cent_x + np.round((ann_x / 2))) - 1):
        for idy in range(int(cent_y - np.round((ann_y / 2))), int(cent_y + np.round((ann_y / 2))) - 1):
          if idx1 < I.shape[1] and idy < I.shape[0] and idx1 >= 0 and idy >= 0:
            if I_ann[indey, index, 0] == 255:
              anns_img2[idy, idx1] = potential_objects[objIdx]
          indey = indey + 1
        index = index + 1
        indey = 0

      ann_area = np.sum(anns_img2 == ann['category_id'])

      # If the annotation is completely hidden, remove the annotation
      if ann_area == 0:
        anns.remove(ann)
      else:  # update bounding box
        [y, x] = np.where(anns_img2 == ann['category_id'])
        anns[ann_idx]['bbox'] = [int(np.min(x)), int(np.min(y)), int((np.max(x) - np.min(x))),
                                 int(np.max(y) - np.min(y))]

        [LE, TE, annWidth, annHeight] = anns[ann_idx]['bbox']
        ann_idx += 1

  # ADD AUGMENTED ANNOTATION INFO
  aug_ann_new = {}
  aug_ann_new['image_id'] = image_id 
  aug_ann_new['id'] = random.randint(100000, 123456789)
  aug_ann_new['area'] = int(aug_area)
  aug_ann_new['segmentation'] = Rs
  aug_ann_new['iscrowd'] = int(0)
  aug_ann_new['category_id'] = int(potential_objects[objIdx])
  aug_ann_new['bbox'] = [int(left_edge), int(top_edge), int(bbox_width), int(bbox_height)]
  anns.append(aug_ann_new)

  return I, anns
