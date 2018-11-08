import os
import math
import sys
import numpy as np
import cv2
import tensorflow as tf
import glob
import time


RESIZE=60
raw_data_dir = ''
output_dir = ''
langs = 'tha+eng'
unichar_dir = output_dir+langs
alphabet = u''

def setDevice(number_str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = number_str

# create the charset
def getAlphabet(unichar_dir):
    num_cat = 2 + ord(u'\u007f')+1-ord(u'\u0021') + ord(u'\u0e7f')+1-ord(u'\u0e00')
    unichar_path = unichar_dir+'/charset_size={}.txt'.format(num_cat)
    index = 0
    if not os.path.exists(unichar_path):
        with open(unichar_path, 'w', encoding='utf-8') as f:
            f.write('0\t \n')
            for i in range(ord(u'\u0021'), ord(u'\u007f')+1):
                index += 1
                f.write(str(index)+'\t'+chr(i)+'\n')
            for i in range(ord(u'\u0e00'), ord(u'\u0e7f')+1):
                index += 1
                f.write(str(index)+'\t'+chr(i)+'\n')
            f.write('{}\t<nul>\n'.format(num_cat-1))
    alphabet = u''
    with open(unichar_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines[:len(lines)-1]:
            alphabet += line.strip('\n').split('\t')[1]
    return alphabet

def text_to_label(text):
    if text=='<nul>':
        return len(alphabet)
    else:
        return alphabet.find(text)

def label_to_text(label):
    if label == len(alphabet):
        return '<nul>'
    else:
        return alphabet[label]

def getBoxes(box_path):
    boxes = []
    with open(box_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip('\n').split(' ')
            if len(info) >= 7:
                info.pop(0)
                info[0] = ' '
            boxes.append(info)
        f.close()
    return boxes

def getBlob(index, images, lines_bounds):
    page_num = lines_bounds[4][index]
    l = lines_bounds[0][index]
    b = lines_bounds[1][index]
    r = lines_bounds[2][index]
    t = lines_bounds[3][index]
    img = images[page_num]
    return img[t:b, l:r, :]

def getLines(boxes, height):
    min_left, min_top = sys.maxsize, sys.maxsize
    max_right, max_bottom = 0, 0
    text = ''
    texts, lefts, bottoms, rights, tops, pages= [], [], [], [], [], []
    for box in boxes:
        # get the min bottom and max top for one line
        char, left, bottom, right, top, page = box
        left = int(left)
        right=int(right)
        # change to upper left origin coordinate
        bottom = height-int(bottom)
        top = height-int(top)
        
        if char != '\t' and box != boxes[-1]: 
            # update boundaries
            text += char
            if min_left > left: min_left = left
            if min_top > top: min_top = top
            if max_right < right: max_right = right
            if max_bottom < bottom: max_bottom = bottom
        else: # EOL
            texts.append(text)
            lefts.append(min_left)
            bottoms.append(max_bottom)
            rights.append(max_right)
            tops.append(min_top)
            pages.append(int(page))
            # reset for next line
            text = ''
            min_left, min_top = sys.maxsize, sys.maxsize
            max_right, max_bottom = 0, 0       
    return [texts, lefts, bottoms, rights, tops, pages]

def readImages(image_path):
    ret, images = cv2.imreadmulti(image_path)
    return images

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def resizeLine(height, image):
    image = np.dot(image[:], [0.299, 0.587, 0.114]) # change to grayscale
    ratio = height/image.shape[0]
    image = cv2.resize(image, (math.floor(ratio*image.shape[1]), height))
    image = image.astype(np.float32)
    return image

def padLabels(label, max_length=200):
    nul_id = text_to_label('<nul>')
    padded = [nul_id]
    for l in label:
        padded.append(l)
        padded.append(nul_id)
    assert len(padded)<=max_length, "max length for padding is too small"
    size = 0
    while len(padded) < max_length:
        padded.append(nul_id)
        size+=1
    return padded

# box file: <0:symbol/tab> <1:left> <2:bottom> <3:right> <4:top> <5:page num>
def createTFRecord(box_path, train_eval):
    basename = os.path.basename(box_path)
    tfrecords_dir = output_dir+train_eval+'/'
    if not os.path.exists(tfrecords_dir): os.mkdir(tfrecords_dir)
    tfrecords_filename = tfrecords_dir+os.path.splitext(basename)[0]+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    boxes = getBoxes(box_path)
    images = readImages(os.path.splitext(box_path)[0]+'.tif')
    height, width, channel = images[0].shape
    print(tfrecords_filename)
    # lines = [labels, lefts, bottoms, rights, tops]
    lines = getLines(boxes, height)
    # write in tfrecord
    for i in range(len(lines[0])):
        text = lines[0][i]
        label = [text_to_label(x) for x in text]
        blob = getBlob(i, images, lines[1:])
        if blob.shape[0]==0:
            continue
        img_resize = resizeLine(RESIZE, blob)
        _height, _width = img_resize.shape
        success, img_raw = cv2.imencode('.png', img_resize)
        img_raw = tf.compat.as_bytes(img_raw.tostring())
        example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image/format': _bytes_feature(tf.compat.as_bytes('PNG')),   
                    'image/encoded': _bytes_feature(img_raw),
                    'image/class': _int64_list_feature(padLabels(label)),
                    'image/unpadded_class': _int64_list_feature(label),
                    'image/width': _int64_feature(_width),
                    'image/height': _int64_feature(_height),
                    'image/text': _bytes_feature(tf.compat.as_bytes(text))
                }))
        writer.write(example.SerializeToString()) 
    
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("box_path", type=str, help="the full path name of the box file")
    parser.add_argument("type", type=str, choices=['train','eval'], help="create data for train or eval")
    parser.add_argument("--data_dir", type=str, default='../dataset/',
			help="raw data directory, default='../dataset/'")
    parser.add_argument("--output_dir", type=str, default='../dataset/tha+eng/tfrecords/', help="directory of output dataset, default='../dataset/tha+eng/tfrecords/'")
    parser.add_argument("--langs", type=str, default='tha+eng', help="languages, default='tha+eng'")
    args = parser.parse_args()
    raw_data_dir = args.data_dir
    output_dir = args.output_dir
    langs = args.langs
    if output_dir == '../dataset/tha+eng/tfrecords/':
        if langs != 'tha+eng':
            output_dir = '../dataset/{}/tfrecords/'.format(langs)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    unichar_dir = raw_data_dir+langs
    alphabet = getAlphabet(unichar_dir)
    # print("finish create tfrecord for",os.path.basename(args.box_path))
    createTFRecord(args.box_path, args.type)
