import sys
from urllib import parse
import os
import json
from multiprocessing import Pool
import html5lib
import time
import re


def get_entity_id_dict(infilename):
    entity2id_dict = {}
    index = 0
    with open(infilename, "r", encoding = "utf-8") as infile:
        for line in infile:
            line_list = line.strip().split()
            if line_list[0] not in entity2id_dict:
                entity2id_dict[line_list[0]] = None
    return entity2id_dict

def generate_alias_entity2index_file(entity2id_dict, alias_entity_file, \
    output_alias_name2uri_file, output_alias_uri2index_file, type_option = ""):
    index = -1
    with open(output_alias_name2uri_file, "w", encoding = "utf-8") as output_name2uri_file:
        with open(output_alias_uri2index_file, "w", encoding = "utf-8") as output_uri2index_file:
            with open(alias_entity_file, "r", encoding = "utf-8") as infile:
                for line in infile:
                    index += 1
                    if index == 0: 
                        continue
                    line_list = line.strip().split()
                    if line_list[0] in entity2id_dict:
                        output_name2uri_file.write("%s\t%s%s\n" % (line_list[0], type_option, str(index)))
                        output_uri2index_file.write("%s%s\t%s\n" % (type_option, str(index), str(index - 1)))



if __name__ == '__main__':
    if len(sys.argv) < 6:
        print ("Usage: python3 aliagn_trained_vecs.py all_entity2id_infilename alias_entity_file output_name2uri_file output_uri2index_file type_option")
        exit(0)
        
    all_entity2id_infilename = sys.argv[1]
    alias_entity_file = sys.argv[2]
    output_name2uri_file = sys.argv[3]
    output_uri2index_file = sys.argv[4]
    type_option = sys.argv[5]
    if type_option == "entity":
        type_option = "Q"
    elif type_option == "image":
        type_option = "I"
    else:
        type_option = ""
    
    all_entity2id_dict = get_entity_id_dict(all_entity2id_infilename)
    generate_alias_entity2index_file(all_entity2id_dict, alias_entity_file, \
        output_name2uri_file, output_uri2index_file, type_option)