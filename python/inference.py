"""Model inference separate from training."""
from tensorflow import app
from tensorflow.python.platform import flags
import os
import tesserocr
import numpy as np
from tesserocr import PyTessBaseAPI, RIL
import shutil
import tempfile
from PIL import Image
import math
import tensorflow as tf
import cv2

import vgsl_model


flags.DEFINE_string('graph_def_file', None,
                    'Output eval graph definition file.')
flags.DEFINE_string('train_dir', '/tmp/mdir',
                    'Directory where to find training checkpoints.')
flags.DEFINE_string('model_str',
                    '1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225',
                    'Network description.')
flags.DEFINE_string('image', None, 'Inference image path')
flags.DEFINE_string('decoder', '../dataset_ctc/tha+eng/charset_size=225.txt', 'Charset decoder')

FLAGS = flags.FLAGS

# initialise params
alphabet=''
RESIZE=60

# extract lines
def getLineImages(image_path):
    image = Image.open(image_path)
    lines = []
    with PyTessBaseAPI() as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        print('Found {} box image components.'.format(len(boxes)))
        for i in range(len(boxes)):
            line = np.asarray(boxes[i][0])
            lines.append(line)
    return lines

# resize line to be the same height
def resizeLine(height, image):
    ratio = height/image.shape[0]
    image = cv2.resize(image, (math.floor(ratio*image.shape[1]), height))
    image = image.astype(np.float32)
    return image

# get unichar and map accordingly
def getAlphabet(unichar_path):
#    num_cat = 2 + ord(u'\u007f')+1-ord(u'\u0021') + ord(u'\u0e7f')+1-ord(u'\u0e00')
#    unichar_path = unichar_dir+'/charset_size={}.txt'.format(num_cat)
#    index = 0
#    if not os.path.exists(unichar_path):
#        with open(unichar_path, 'w', encoding='utf-8') as f:
#            f.write('0\t \n')
#            for i in range(ord(u'\u0021'), ord(u'\u007f')+1):
#                index += 1
#                f.write(str(index)+'\t'+chr(i)+'\n')
#            for i in range(ord(u'\u0e00'), ord(u'\u0e7f')+1):
#                index += 1
#                f.write(str(index)+'\t'+chr(i)+'\n')
#            f.write('{}\t<nul>\n'.format(num_cat-1))
    alphabet = u''
    with open(unichar_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines[:len(lines)-1]:
            alphabet += line.strip('\n').split('\t')[1]
    return alphabet, unichar_path

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

def cleanup(temp_dir):
    ''' Tries to remove the whole temp_dir '''
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


# create tfrecord for inference
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createTFRecord(image_path, tempdir):
    # list of nparray of line images
    lines = getLineImages(image_path)
    # create a temp image.tfrecords
    basename = os.path.basename(image_path)
    tfrecords_filename = tempdir+'/'+str(len(lines))+'-'+os.path.splitext(basename)[0]+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    # write in tfrecord
    for i in range(len(lines)):
        text = ''
        label = [text_to_label(x) for x in text]
        blob = lines[i]
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
                    'image/unpadded_class': _int64_list_feature(label),
                    'image/width': _int64_feature(_width),
                    'image/height': _int64_feature(_height),
                    'image/text': _bytes_feature(tf.compat.as_bytes(text))
                }))
        writer.write(example.SerializeToString())

    writer.close()
    return tfrecords_filename

def setup(encoding="UTF-8",  cuda_device=""):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["PYTHONIOENCODING"] = encoding


def main(argv):
  del argv
  alphabet = getAlphabet(FLAGS.decoder)
  tempdir = tempfile.mkdtemp()
  infer_data = createTFRecord(FLAGS.image, tempdir)
  num_lines = int(os.path.basename(infer_data).split('-', 1)[0])
  setup()
  import time
  start = time.time()
  vgsl_model.Inference(FLAGS.train_dir, FLAGS.model_str,
                  infer_data, FLAGS.decoder, num_lines,
                  FLAGS.graph_def_file)
  print('spent: {:.2f}s'.format(time.time()-start))
  cleanup(tempdir)

if __name__ == '__main__':
  app.run()

