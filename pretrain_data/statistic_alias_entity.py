from bs4 import BeautifulSoup
import requests
import sys
from urllib import parse
import os
import json
from multiprocessing import Pool
import html5lib

# input_folder = "pretrain_data/output"
#input_folder = "../pretrain_data/entity_occurred"
#output_folder = "../pretrain_data/entity_alias"

def get_file_list(input_folder):
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    print ("file_list=%s" % file_list)
    return file_list

def run_proc(idx, n, file_list, input_folder, output_folder):
    for i in range(len(file_list)):
        if i % n == idx:
            input_name = file_list[i]
            print("input_name=%s" % input_name)
            target = input_name.replace(input_folder, output_folder)
            folder = '/'.join(target.split('/')[:-1])
            print ("target=%s" % target)
            if not os.path.exists(folder):
                os.makedirs(folder)

            fout = open(target, "w")

            with open(input_name, "r") as infile:
                for lines in infile:
                    fout.write("%s\n" % "\n".join(lines.split("\t")))

            fout.close()


import sys

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ("Usage: python statistic_alias_entity.py process_num input_folder output_folder")
        exit(0)
        
    n = int(sys.argv[1])
    input_folder = sys.argv[2] + "/"
    output_folder = sys.argv[3] + "/"
    
    file_list = get_file_list(input_folder)
    
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i,n, file_list, input_folder, output_folder))
    p.close()
    p.join()