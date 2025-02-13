## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 3.4+
- PyTorch 1.0
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install mmdetection

a. Install PyTorch 1.0 and torchvision following the [official instructions](https://pytorch.org/).

b. Download this code repository.

c. Compile cuda extensions.

```shell
cd semantic-augmentation
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

d. Install mmdetection (other dependencies will be installed automatically).

```shell
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

Note: You need to run the last step each time you pull updates from github.
The git commit id will be written to the version number and also saved in trained models.

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$semantic-augmentation/data`.

```
semantic-augmentation
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### Prepare glove embeddings

Download the [glove embeddings](https://nlp.stanford.edu/projects/glove/), and symlink the dataset root to `$semantic-augmentation/mmdet/datasets`. For reference, the embeddings used in the paper are from the Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip.

### Create object bank

Run the following script.

```shell
python create_bank.py
```


### Scripts
Just for reference, [Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Notice
You can run `python(3) setup.py develop` or `pip install -e .` to install mmdetection if you want to make modifications to it frequently.

If there are more than one mmdetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
