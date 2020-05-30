from bs4 import BeautifulSoup
import sys
from urllib import parse
import os
from multiprocessing import Pool

# input_folder = "pretrain_data/output_2"
input_folder = "../pretrain_data/output"
output_folder = "../pretrain_data/ann"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print (file_list)

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

            soup = BeautifulSoup(open(input_name), features="html5lib")
            print ("soup=%s" % soup)
            docs = soup.find_all('doc')
            print ("doc=%s" % docs)


            fout = open(target, "w")

            for doc in docs:
                doc_text = doc.get_text(" sepsepsep ").split("[_cutcutcut_]")
                content = doc_text[0]
                wordsegment = doc_text[1]
                print ("original content=%s" % content)
                while content[0] == "\n":
                    content = content[1:]
                print("input content=%s" % content)
                content = [x.strip() for x in content.split("\n")]
                print("second content=%s" % content)
                # begin with 1 to filter article title(located in the 1st line)
                # if have no title, use "content[0:]"
                content = "".join(content[1:])
                print ("normalize content=%s" % content)

                lookup = [(x.get_text().strip(), parse.unquote(x.get('href'))) for x in doc.find_all("a")]
                lookup = "[_end_]".join(["[_map_]".join(x) for x in lookup])
                print ("lookup=%s[_end_]%s" % (content, lookup))
                fout.write(content+"[_end_]"+lookup+"[_cutcutcut_]"+wordsegment+"\n")

            fout.close()

import sys

n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i,n, file_list))
p.close()
p.join()
