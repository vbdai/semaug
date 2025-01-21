from .xml_style import XMLDataset
import numpy as np


class VOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        elif 'VOC2021' in self.img_prefix:
            self.year = 2021
        elif 'VOC2022' in self.img_prefix:
            self.year = 2022
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
            
        words = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
             'dog', 'horse', 'motorcycle', 'person', 'plant', 'sheep', 'sofa', 'train', 'tv']
        
        self.embeddings_index = {}
        self.track = np.zeros([21])
        path_to_glove_file = self.path_to_glove_file #"/home/work/user-job-dir/COCP-main3/mmdet/datasets/glove.6B.300d.txt"
        if 'txt' in path_to_glove_file:
            with open(path_to_glove_file, encoding="utf8") as f2:
                for line in f2:
                    word, coefs = line.split(maxsplit=1)
                    if word in words:
                        coefs = np.fromstring(coefs, "f", sep=" ")
                        self.embeddings_index[word] = coefs
        
        self.d = {}
        d_object = {}
        with open("mmdet/datasets/labels_voc.txt") as f1:
            for line in f1:
                (key, val) = line.split(': ')
                self.d[int(key)] = val.strip('\n')
                # print(val)
                if int(key) < 91:
                    # print(val)
                    d_object[int(key)] = val.strip('\n')
    
       
        self.obj_class_names = list(d_object.values())
