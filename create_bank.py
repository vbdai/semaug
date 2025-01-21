import matplotlib
# matplotlib.use('TkAgg')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import urllib.request as req

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# from urllib.error import HTTPError

# auth = req.HTTPBasicAuthHandler()
# opener = req.build_opener(proxy, auth, req.HTTPHandler)
# req.install_opener(opener)
# conn = req.urlopen('http://google.com')
# return_str = conn.read()

dataDir = 'data/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

ids = [cat['id'] for cat in cats]

for catidx in range(0, np.max(ids)+1):# get all images containing given categories, select one at random
    try:
        # catIds = coco.getCatIds(catNms=['toaster']);
        imgIds = coco.getImgIds(catIds=catidx)

        dirName = os.path.join("data/cocoBank", str(catidx))

        try:
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")

        for idx in range(0, len(imgIds)):
            # imgIds = coco.getImgIds(imgIds = [324158])
            img = coco.loadImgs(imgIds[idx])[0]

            # load and display image
            I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
            # use url to load image
            # try:
            # I = io.imread(img['coco_url'])
            # except HTTPError as e:
            #     content = e.read()
            # matplotlib.axis('off')
            # matplotlib.pyplot.imshow(I)
            # matplotlib.show()
            #
            # load and display instance annotations
            # matplotlib.pyplot.imshow(I); #matplotlib.axis('off')
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catidx, iscrowd=None)
            anns = coco.loadAnns(annIds)
            # coco.showAnns(anns)

            mask = np.zeros((img['height'], img['width']))
            # coco.annToMask(anns[0])
            for i in range(len(anns)):
                try:
                    mask = np.zeros((img['height'], img['width']))
                    mask += coco.annToMask(anns[i]) / np.max(coco.annToMask(anns[i]))
                    # mask.dtype(np.int)
                    x1 = np.int(np.round(anns[i]['bbox'][0]))
                    x2 = np.int(np.round(anns[i]['bbox'][0]) + np.round(anns[i]['bbox'][2]))
                    y1 = np.int(np.round(anns[i]['bbox'][1]))
                    y2 = np.int(np.round(anns[i]['bbox'][1]) + np.round(anns[i]['bbox'][3]))
                    # plt.figure()
                    import scipy.misc

                    io.imsave(os.path.join(dirName, (str(imgIds[idx]) + '_' + str(i) + '_ann.png')), mask[y1:y2, x1:x2])
                    # plt.imshow(mask[y1:y2,x1:x2])
                    # plt.figure()
                    # plt.imshow(I[y1:y2, x1:x2])
                    io.imsave(os.path.join(dirName, (str(imgIds[idx]) + '_' + str(i) + '.png')), I[y1:y2, x1:x2])
                except:
                    print("Something went wrong")

    except:
        print("Something went wrong")
