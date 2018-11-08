import os
import random
import linecache
import subprocess 
import shlex
import re
import tempfile
import time
import ntpath
import sys

output_dir = '../dataset/'
fonts_dir = 'usr/share/fonts'

max_num_chars = 100 # max number of chars for 1 piece of language
# txt2img_command = '/workspace/tesseract-ocr/src/training/.libs/lt-text2image'
txt2img_command = 'text2image'


def getAllFilePaths(data_path, extension):
    import os.path
    paths = []
    for f in os.listdir(data_path):
        ext = os.path.splitext(f)[1]
        #print(ext)
        if ext.lower() != extension:
            continue
        paths.append(os.path.join(data_path, f))
    return paths

class TesseractError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)
        
def get_errors(error_string):
    return u' '.join(
        line for line in error_string.decode('utf-8').splitlines()
    ).strip()

# generate tiff/box pair from a file
# size of the image default: 1500*2000
def getTifBox(langs, input_filename, outputbase, fontname, xsize = 1500, ysize = 1000):
    command = []
    
    command += (txt2img_command, '--text', input_filename,
                '--outputbase', outputbase)
    command += ('--xsize={}'.format(xsize), '--ysize={}'.format(ysize))
    command += ('--font', fontname)
    command += ('--fonts_dir', fonts_dir)
    config = ' --ptsize=10'
    command += shlex.split(config)
    #print(command)
    proc = subprocess.Popen(command, stderr=subprocess.PIPE)
    #print("pass proc")
    status_code, error_string = proc.wait(), proc.stderr.read()
    proc.stderr.close()

    if status_code:
        raise TesseractError(status_code, get_errors(error_string))

def renderText(langs, path_to_text, outputbase, font):
    if not os.path.exists(outputbase):
        os.makedirs(outputbase) 
    
    file_name = ntpath.basename(path_to_text).split('.')[0]
    font = font.strip('"')
    getTifBox(langs, path_to_text, 
              outputbase+file_name+'-'+font,font)

def main(langs, txt_file_name, font):
    start_time = time.time()
    txt_file_name = os.path.basename(txt_file_name)
    path_to_text = output_dir+langs+'/txt/'+txt_file_name
    path_to_final = output_dir+langs+'/boxtif/'
    renderText(langs, path_to_text, path_to_final, font)
    print('---finish process for ', txt_file_name,font, ' ----time spent', 
          round(time.time()-start_time,2))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("langs", type=str, help="languages separated by +, e.g. tha+eng")
    p.add_argument("txt", type=str, help="the basename of the input txt source file, e.g. 0.txt")
    p.add_argument("font", type=str, help="font name, e.g. \"Garuda\"")
    p.add_argument("--fonts_dir", type=str, default="usr/share/fonts", help="the directory of the fonts to use, default=/usr/share/fonts")
    p.add_argument("--input_dir", type=str, default="../dataset/", help="dataset path of input text, defualt=\"../dataset/\"")
    args = p.parse_args()
    fonts_dir = args.fonts_dir
    output_dir = args.input_dir
    main(args.langs, args.txt, args.font)
