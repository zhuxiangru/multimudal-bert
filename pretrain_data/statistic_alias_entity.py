from bs4 import BeautifulSoup
import requests
import sys
from urllib import parse
import os
import json
from multiprocessing import Pool
import html5lib

# input_folder = "pretrain_data/output"
input_folder = "../pretrain_data/entity_occurred"
output_folder = "../pretrain_data/entity_alias"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print ("file_list=%s" % file_list)

def run_proc(idx, n, file_list):
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
    n = int(sys.argv[1])
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i,n, file_list))
    p.close()
    p.join()