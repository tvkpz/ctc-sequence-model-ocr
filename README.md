# Multilingual Sequence Model.

Refer to 
A TensorFlow implementation of the STREET model described in the paper:

"End-to-End Interpretation of the French Street Name Signs Dataset"

Available at: http://link.springer.com/chapter/10.1007%2F978-3-319-46604-0_30

The code has been mainly referred to the [STREET model](https://github.com/mldbai/tensorflow-models/tree/master/street).

See [inference output](python/inference.ipynb) for ocr example.

## Contents
* [Introduction](#introduction)
* [Installing and setting up the sequence model](#installing-and-setting-up-the-street-model)
* [Create dataset](#create-dataset)
* [Training a full model with evaluation](#training-a-full-model-with-evaluation)
* [Inference and Comparison with Tesseract performance](#inference-and-omparison-with-tesseract-performance)
* [The Variable Graph Specification Language](#the-variable-graph-specification-language)

## Introduction

The model trains both Thai and English in one charset. The input is an textline image, and the output is a sequence of text.


## Installing and setting up the sequence model
```
git clone https://github.wdf.sap.corp/I351756/sequence-model.git
```

### Build from docker file. 
Create docker image from docerfile folder.
```
cd dockerfile
docker build -t <image name> .
```
Create container using 
```
nvidia-smi docker run -it -d --name <container name> -p 8888:8888 -p 6006:6006 -v <directory to be mapped>:<target mapped directory in docker> <image name>  /bin/bash
```
### Build from scratch
- Build an container from docker image `tensorflow/tensorflow:latest-gpu-py3` or `tensorflow/tensorflow:latest-py3`(Tested for Tensorflow/Tensorflow-gui 1.7 and 1.9)
- Set all proxy (refer to `dockerfile/Dockerfile`)
- Install some python dependencies
```
pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        keras \
        nltk \
	scikit-image \
        && \
    python3 -m ipykernel.kernelspec
```
- Install opencv
```
bash dockerfile/install-opencv.sh
```

### Build some dependencies
tesseract 4.0 beta3. Refer to https://github.com/Layneww/Tesseract-Notes/blob/master/setupTesseract.md, Build Tesseract master branch from source. Please also build the training.

tesserocr: the python wrapper for tesseract. (optional, install for inference)
```
# go into the tesseract directory just git clone from github
cd tesseract
# include 'tesseract/osdetect.h' in library
cp src/ccmain/osdetect.h /usr/local/include/tesseract/
# install tesserocr 
pip install tesserocr
```
pytesseract: another python wrapper (optional, install for inference)
```pip install pytesseract```

### Build ops
Go into the directory of this project.
```
cd cc
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared rnn_ops.cc -o rnn_ops.so -fPIC -I $TF_INC -O3 -mavx -D_GLIBCXX_USE_CXX11_ABI=0 -L $TF_LIB -ltensorflow_framework -O2
```
If appearing ```fatal error: nsync_cv.h: No such a file or directory```, add ```-I$TF_INC/external/nsync/public``` to the g++ command.
Refering to [TensorFlow Instruction](https://www.tensorflow.org/extend/adding_an_op#build_the_op_library) for more details.

Run the unittests:
```
cd python
python3 decoder_test.py
python3 errorcounter_test.py
python3 shapes_test.py
python3 vgslspecs_test.py
python3 vgsl_model_test.py
```

## Create dataset

### Create text files
You can other download the text files from the release or create own customised text files.
#### Download from the release
10k text files. One txt contains 4 snetences, 2 from English and Thai each. Each sentence is no more than 100 characters.
Put the upzipped dataset folder under the main diretory of this project.
#### Generate own text files 
To create the similar text files described above, 
```
cd python
mkdir ../dataset
python3 getText.py -d DATA_PATH tha+eng 2+2 10000
```
```DATA_PATH``` is the path of the corpus data e.g. W2C. Get more information using ```-h``` flag.

Files are created like
```
dataset/
  tha+eng/
    tha+eng.count.txt
    txt/
      0.txt
      1.txt
      ...
```
#### Generate Tif/box files from the text file
Basic operation: ```python3 getTifBox.py [--fonts_dir FONTS_DIR] [--input_dir INPUT_DIR] langs txt font```
e.g. ```python3 getTifBox.py --fonts_dir ../thafonts tha+eng 0.txt Garuda```
Use parallel program to speed up the generation. (please first install GNU parallel by ```apt-get install parallel```)
```
files=`ls ..dataset/tha+eng/txt/*`
fonts_dir=../thafonts
parallel -j40 python3 getTifBox.py --fonts_dir $fonts_dir tha+eng ::: `basename -a $files` ::: `cat $fonts_dir/tha.fontlist.txt`
```
The box/tif files will be put in ```dataset/tha+eng/boxtif/```

#### Generate tfrecords
Basic operation: ```python3 create_tfrecords.py [--data_dir DATA_DIR]
                           [--output_dir OUTPUT_DIR] [--langs LANGS]
                           box_path {train,eval}```
e.g. ```python3 create_tfrecords.py --langs tha+eng
                           ../dataset/tha+eng/boxtif/0-Garuda.box train```
It will also create the decoder file whose path is ```dataset/tha+eng/charset_size=225.txt```
Use parallel prgram, 
firstly, write 2 txt files called `train.txt` `eval.txt` in `tha+eng` directory, including the full path of box files in each trainset and evalset.

```
cat ../dataset/tha+eng/train.txt | parallel -j40 create_tfrecords.py --langs tha+eng {} train
cat ../dataset/tha+eng/eval.txt | parallel -j40 create_tfrecords.py --langs tha+eng {} eval
```

The tfrecords will be put in ```dataset/tha+eng/tfrecords/```, consisting of 2 subfolder `train/` and `eval/`.

## Training a full model with evaluation
```
cd python
train_dir=../model
# rm -rf $train_dir  # uncomment this if want to retrain the whole model
CUDA_VISIBLE_DEVICES="0" python3 vgsl_train.py -model_str='1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225' --train_data=../dataset/tha+eng/tfrecords/train/* --train_dir=$train_dir --max_steps=1000000 &
CUDA_VISIBLE_DEVICES="" python3 vgsl_eval.py --model_str='1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225' --num_steps=1000 --eval_data=../dataset/tha+eng/tfrecords/eval/*  --decoder=../dataset/tha+eng/charset_size=225.txt --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=$train_dir
```

## Inference and Comparison with Tesseract performance
### Inference of the sequence model
To test the sequence model. Suppose we have an image called `test.png` mixed with thai and english,
get the OCR result
```
export PYTHONIOENCODING=UTF-8
train_dir=../model
python3 inference.py --train_dir=$train_dir --model_str='1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225'
                    --image=test.png --decoder=../dataset/tha+eng/charset_size=225.txt
```
The output will first print how many lines it finds during the preprocessing, and output the OCR result.
You can use a pre-trained model, which can be downloaded in the release (`model.zip`). Then just set the `--train_dir` to the path of the model.
### Inference of tesseract
```
tesseract -l tha+eng test.png stdout --tessdata <the path of the tessdata folder>
```
### Jupyter notebook
You may also refer to `python/inference.ipynb`.

### Test images
Some example images under the `test-image.zip` can be downloaded in the release.

## The Variable Graph Specification Language
Please see https://github.com/mldbai/tensorflow-models/blob/master/street/g3doc/vgslspecs.md for more details.
