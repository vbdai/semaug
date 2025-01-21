# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data loader and processing."""
# from absl import logging
# import tensorflow as tf
import torch

# import utils
# from keras import anchors
# from object_detection import preprocessor
# from object_detection import tf_example_decoder_seg as tf_example_decoder
import torch.nn.functional as f

import random
import os
# from tensorflow.python.keras import backend as K
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


# import io

def semaug_proc(image_id, anns, I, track, embeddings_index, d, obj_class_names,
                embedding_dim): 

    if not anns: return I, anns

    # Create a list of the segmentations
    seglist = []
    seglist_obj = []
    

    for i in range(len(anns['labels'])):
        if anns['labels'][i] > 0:
            seglist_obj.append(d[anns['labels'][i]])
            seglist.append(d[anns['labels'][i]])
        

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
        # stuffnum = [k for (k, v) in d.items() if v == word]
        if word == 'potted plant':
            word = 'plant'
        elif word =='aeroplane':
            word = 'airplane'
        elif word == 'diningtable':
            word = 'table'
        elif word == 'motorbike':
            word = 'motorcycle'
        elif word == 'pottedplant':
            word = 'plant'
        elif word == 'tvmonitor':
            word = 'tv'
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
        elif word == 'unlabeled':
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # if stuffnum:
                # if int(stuffnum[0]) > 90:
                    # logging.info("EMBEDDING STUFF IS SOMETHING")
            embedding_matrix_sample_stuff[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            print(word)
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
        elif word == 'unlabeled':
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
        # print(idx_obj, word)
        if word not in input_words:
            # print("word", word, "is not in input_words")
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
            elif word == 'unlabeled':
                continue
            embedding_vector = embeddings_index.get(word)
            # print("embedding vector is", embedding_vector)
            if embedding_vector is not None:
                embedding_matrix_objects[idx_obj] = embedding_vector
                embedding_matrix_objects_cos[idx_obj] = embedding_vector
                hits += 1
            else:
                misses += 1
                print("WORD IS MISSING:", word)

   
    # Find the location of object placement through cosine similarity
    embedding_matrix_objects_cos = torch.from_numpy(embedding_matrix_objects_cos)
    embedding_matrix_sample = torch.from_numpy(embedding_matrix_sample)
    normed_embedding = f.normalize(embedding_matrix_objects_cos, dim=1,
                                   p=2)  # tf.nn.l2_normalize(embedding_matrix_objects_cos)
    normed_array = f.normalize(embedding_matrix_sample, dim=1, p=2)  # tf.nn.l2_normalize(embedding_matrix_sample)

    

    cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
    closest_cosine_words = torch.topk(cosine_similarity, k=3)  # tf.nn.top_k(cosine_similarity, k=3)
    closest_cosine_words = closest_cosine_words.indices[0].numpy()
    potential_objects = []

    for word_idx in range(0, 3):
        # whatword = obj_class_names[closest_k_words[word_idx]]
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

    # logging.info(potential_objects)
    objectList = []
    for i in range(0, len(potential_objects)):
        objectList.append(track[potential_objects[i]])
    # objectList = [track[potential_objects[0]], track[potential_objects[1]], track[potential_objects[2]]]  # ,
    objIdx = objectList.index(np.min(objectList))
    track[potential_objects[objIdx]] += 1

    # Find the embedding vector for the object to be pasted
    word = obj_class_names[closest_cosine_words[objIdx]]  # obj_class_names[ran_obj]
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
    normed_embedding = f.normalize(embedding_matrix_sample_stuff, dim=-1,
                                   p=2)  # tf.nn.l2_normalize(embedding_matrix_objects_cos)
    normed_array = f.normalize(embedding_vector.reshape((1, len(embedding_vector))), dim=-1,
                               p=2)  # tf.nn.l2_normalize(embedding_matrix_sample)

    cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
    closest_k_words = torch.topk(cosine_similarity, k=1)
    closest_k_words = closest_k_words.indices[0].numpy()

    whereword = [k for (k, v) in embeddings_index.items() if
                 np.allclose(v, embedding_matrix_sample_stuff[closest_k_words[0]])]

    if whereword:
        wherenum = [idx for idx, key in enumerate(list(d.items())) if
                        whereword[0] in key[1]]
    else:
        # Find the location of object placement through cosine similarity
        normed_embedding = f.normalize(embedding_matrix_sample, dim=1,
                                       p=2)  # tf.nn.l2_normalize(embedding_matrix_objects_cos)
        normed_array = f.normalize(embedding_vector.reshape((1, len(embedding_vector))), dim=1,
                                   p=2)  # tf.nn.l2_normalize(embedding_matrix_sample)

        cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
        closest_k_words = torch.topk(cosine_similarity, k=1)
        closest_k_words = closest_k_words.indices[0].numpy()

        whereword = [k for (k, v) in embeddings_index.items() if
                     np.allclose(v, embedding_matrix_sample[closest_k_words[0]])]

        wherenum = [idx for idx, key in enumerate(list(d.items())) if
                        whereword[0] in key[1]]


    # Get the center coordinates for the object placement
    cent_x = 0
    cent_y = 0
    # If the location to paste is in the "stuff" category
    if int(wherenum[0]) > 91:
        if cent_x == 0:
            for i in range(0, len(anns)):
                if anns['labels'][i] in wherenum:
                    cent_x = anns['bboxes'][i][0] + (anns['bboxes'][i][2] / 2)
                    cent_y = anns['bboxes'][i][1] + (anns['bboxes'][i][3] / 2)
    else:  # If the location to paste is in the "object" category
        for i in range(0, len(anns['labels'])):
            if anns['labels'][i] in wherenum:
                # print(anns[i]['category_id'])
                cent_x = anns['bboxes'][i][0] + (anns['bboxes'][i][2] / 2)
                cent_y = anns['bboxes'][i][1] + (anns['bboxes'][i][3] / 2)

    if cent_x > I.shape[1]:
        cent_x = np.round(I.shape[1]/2)
    if cent_y > I.shape[0]:
        cent_y = np.round(I.shape[0] / 2)

    # Find the filenames of the objects in the bank and pick one randomly
    dirname = "PascalBank/"
    files = glob.glob(
        os.path.join(dirname, str(potential_objects[objIdx]), '*_ann.png'))

    area_ann = 0
    # Ensure the object to be placed is <300 pixels in area - keeps unrecognizable objects from being placed
    while area_ann < 300:
        try:
            rannum = random.randint(0, len(files) - 1)
            I_ann = mmcv.imread(files[rannum])  # io.imread(files[rannum])
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


    # Paste new object into image and into mask image
    [ann_y, ann_x, channels] = I_ann.shape
    index = 0
    indey = 0
    aug_area = 0
    I_og = I
    for idx1 in range(int(cent_x - np.round((ann_x / 2))), int(cent_x + np.round((ann_x / 2))) - 1):
        for idy in range(int(cent_y - np.round((ann_y / 2))), int(cent_y + np.round((ann_y / 2))) - 1):
            if idx1 < I.shape[1] and idy < I.shape[0] and idx1 >= 0 and idy >= 0:
                if I_ann[indey, index, 0] == 255:
                    I[idy, idx1] = I_annI[indey, index]
                    aug_area += 1
            indey = indey + 1
        index = index + 1
        indey = 0


    # Find new bounding box dimensions
    left_edge = np.maximum(0, int(cent_x - np.round((ann_x / 2))))
    right_edge = np.minimum(I.shape[1], int(cent_x + np.round((ann_x / 2))) - 1)
    top_edge = np.maximum(0, int(cent_y - np.round((ann_y / 2))))
    bottom_edge = np.minimum(I.shape[0], int(cent_y + np.round((ann_y / 2))) - 1)
    bbox_width = right_edge - left_edge - 1
    bbox_height = bottom_edge - top_edge - 1

    anns['labels'] = np.append(anns['labels'],int(potential_objects[objIdx]))
    anns['bboxes'] = np.vstack(
        (anns['bboxes'], np.array([int(left_edge), int(top_edge), int(right_edge), int(bottom_edge)])))
    anns['bboxes'] = anns['bboxes'].astype('float32')

    return I, anns
