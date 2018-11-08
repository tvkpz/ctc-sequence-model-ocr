#!/usr/bin/env python

import os
import random
import linecache
import subprocess 
import shlex
import re
import tempfile

data_path = '/data/DATA/'
output_dir = '../dataset/'
max_num_chars = 100 # max number of chars for 1 piece of language

# get a list of full paths for given extension
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

# split string with multiple delimiters
def tsplit(s, sep):
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))
        stack = pieces
    return stack

# limit the length of aline to be within max_num_chars
def limitSentenceLength(aline):
    orig = aline
    if len(aline) > max_num_chars:
        choices = tsplit(aline, '.!?')
        while aline == '': aline = random.choice(choices)
        #if aline=='': print('.!? empty\n',orig)
        if len(aline) > max_num_chars:
            choices = aline.split(',')
            while aline == '': aline = random.choice(choices)
            if len(aline) > max_num_chars:
                words = aline.split(' ')
                aline=''
                for i in range(len(words)):
                    pre_aline = aline
                    aline += (words[i]+' ')
                    if len(aline) > max_num_chars:
                        aline = pre_aline
                        break
    return aline

# clean the text line to take only tha + eng language ### hard code function
def cleanLine(aline):
    clean_line = []
    for word in aline.split():
        maxchar = max(word)
        if u'\u0020' <= maxchar <= u'\u007f' or u'\u0e00' <= maxchar <= u'\u0e7f':
             clean_line.append(word)
    return ' '.join(clean_line)

# take in a string of languages (lang1+lang2+...)
# return a list of the number of lines for each language
def countNumLines(langs):
    path = output_dir+langs+'/'+langs+'.count.txt'
    count = []
    counted = False
    if os.path.isfile(path):
        if os.path.getsize(path)!=0:
            counted = True
    
    if counted:
        count = [int(x) for x in open(path,'r').readline().split(' ')]
    else:
        lang_list = langs.split('+')
        files = [(data_path+'{}.txt'.format(lang)) for lang in lang_list]
        n_langs = len(lang_list)
        count = [len(open(files[i], 'r', encoding='utf-8').readlines()) 
                 for i in range(n_langs)]
       
        with open(path, 'w') as f:
            count_str = [str(x) for x in count]
            f.write(' '.join(count_str))
    return count

# create raw text files
# if setting create_new==1, it will rewrite all the text files
def createRawData(langs, num_lines_each_lang, num_txt_created, create_new): 
    outputbase = output_dir+langs+'/txt'
    lang_list = langs.split('+')
    n_langs = len(lang_list)
    exist_ntxt = 0 # the num of txt files created under the langs dir
    
    if not os.path.exists(outputbase):
        os.makedirs(outputbase) 
    else:
        print('output data path exists.')
    
    if os.listdir(outputbase)=='' or create_new:
        print('create new raw txt files...')
    else:
        for file in os.listdir(outputbase):
            if file.endswith(".txt"):
                exist_ntxt += 1
        print('create more raw txt files...')
    
    files = [(data_path+'{}.txt'.format(lang)) for lang in lang_list]
    count = countNumLines(langs)
    
    for i in range(exist_ntxt, exist_ntxt+num_txt_created):
        txt_file_path = outputbase+'/{}.txt'.format(i)
        txt_f = open(txt_file_path, 'w', encoding='utf-8')
        
        lines = []
        for j in range(n_langs):
            for k in range(num_lines_each_lang[j]):
                # get a line of example randomly
                aline = linecache.getline(files[j], random.randint(0, count[j])).strip('\n')
                # clean irrevelent language
                aline = cleanLine(aline)
		# clean websites
                aline = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', '', 
                               aline, flags=re.MULTILINE)
                aline = limitSentenceLength(aline) #+'\n'
                lines.append(aline)
        
        # shuffle the lines and store txt and lang code
        random.shuffle(lines)
        txt_f.write(''.join(lines))
        txt_f.close()
        
        # process monitor
        if (i+1)%1000 == 0: print('finish create {} raw text files'.format(i+1))


#def main(langs, num_text, create_new):
#    num_sentences = [2]*len(langs.split('+'))
#    createRawData(langs, num_sentences, num_text, create_new)

if __name__=="__main__":
   # import sys
   # main(sys.argv[1], int(sys.argv[2]),int(sys.argv[3]))
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("langs", type=str, help="languages to write in text files, separated by +, e.g. tha+eng")
    p.add_argument("num_sentences", type=str, 
                   help="numbers of sentences per language in one txt file, same sequence as langs, e.g. 2+2")
    p.add_argument("num_text", type=int, help="number of text files you want to create")
    p.add_argument("-n", "--new", type=int, choices=[0, 1], 
                   help="0 if do not rewrite the previous created txt files",
                   default=1)
    p.add_argument("-d", "--data_path", type=str, default="/data/DATA/",
                   help="path to the language dataset like W2C. default: /data/DATA/")
    args = p.parse_args()
    data_path = args.data_path
    list_num_sentences = [int(x) for x in args.num_sentences.split('+')]
    createRawData(args.langs, list_num_sentences, args.num_text, args.new)
