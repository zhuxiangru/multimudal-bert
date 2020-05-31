import sys
import os
from multiprocessing import Pool
import math
import subprocess
import logging

#input_folder = "../pretrain_data/raw"
#output_folder = "../pretrain_data/data"


def get_file_list(input_folder):
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    file_list = list(set(["_".join(x.split("_")[:-1]) for x in file_list]))
    return file_list

def run_proc(idx, n, file_list, input_folder, output_folder, run_py_file, vocab_file, entity2id_file, image2id_file):
    for i in range(len(file_list)):
        if i % n == idx:
            target = file_list[i].replace(input_folder, output_folder)
            print(file_list[i])
            print(target)
            #command = "python3 ../code/create_instances.py --input_file_prefix {} --output_file {} --vocab_file config_data/vocab.txt --dupe_factor 1 --max_seq_length 256 --max_predictions_per_seq 40"
            #subprocess.run(command.format(file_list[i], target).split())
            command = "python3 {} --input_file_prefix {} --output_file {} --vocab_file {} "\
                "--entity2id_file {} --image2id_file {} "\
                "--dupe_factor 1 --max_seq_length 256 --max_predictions_per_seq 40".format(run_py_file, file_list[i], target, vocab_file, entity2id_file, image2id_file)
            print (command)
            subprocess.run(command.split())


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print ("Usage: python3 create_insts.py process_num input_folder output_folder run_python_file vocab_file entity2id_file image2id_file")
        exit(0)
        
    n = int(sys.argv[1])
    input_folder = sys.argv[2] + "/"
    output_folder = sys.argv[3] + "/"
    run_py_file = sys.argv[4]
    vocab_file = sys.argv[5]
    entity2id_file = sys.argv[6]
    image2id_file = sys.argv[7]
    
    if not os.path.exists(input_folder):
        logging.error("input_folder doesn't exist. input_folder=%s pwd=%s" % (input_folder, os.path.abspath(__file__)))
        exit(0)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_list = get_file_list(input_folder)
        
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i, n, file_list, input_folder, output_folder, run_py_file, vocab_file, entity2id_file, image2id_file))
    p.close()
    p.join()