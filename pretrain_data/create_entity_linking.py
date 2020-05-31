from bs4 import BeautifulSoup
import requests
import sys
from urllib import parse
import os
import json
from multiprocessing import Pool
import html5lib
import time
import re

# input_folder = "pretrain_data/output"
# input_folder = "../pretrain_data/opencc"
#input_folder = "../pretrain_data/opencc_simple"
#output_folder = "../pretrain_data/output"
#entity_occurred_folder = "../pretrain_data/entity_occurred"
MAX_CONTENT_LENGTH = 200

def get_file_list(input_folder):
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    print ("file_list=%s" % file_list)
    return file_list

def entity_linking(text):
    #text = '红楼梦是我国四大名著之一，是小学生的推荐书目。'
    url = r'http://shuyantech.com/api/entitylinking/cutsegment?q=%s&apikey=d07a735852f46f9fe3eff2975dbcae4c' % text
    print ("url=%s" % url)
    try:
        pget = requests.get(url)
        if pget.status_code == 200 or pget.ok == True:
            result = pget.text
        else:
            # 执行不正确则抛弃
            print ("warning: pget.status_code is not 200 and pget.ok is not True. "
                   "So throw this sentence=%s. \nnext" % text)
            result = "{}"
        print ("entity_linking=%s" % result)
    except requests.exceptions.ConnectionError:
        print ("requests.exceptions.ConnectionError, please close the VPNs")
        exit(0)
    return json.loads(result)

def cut_sub_sentences(sentences, max_content_length = MAX_CONTENT_LENGTH):
    # reference: Python中文文本分句 https://www.pythonheidong.com/blog/article/53241/
    sentences = re.sub('([>〉》)）\]】][、])([<〈《(（\[【])', r"\1\n\2", sentences)  # 顿号分隔的括号分隔符
    sentences = re.sub('([>〉》)）\]】])([<〈《(（\[【])', r"\1\n\2", sentences)  # 连在一起的括号分隔符
    sentences = re.sub('([;；:：])(^;；:：)', r"\1\n\2", sentences)  # 分号分隔符
    sentences = re.sub('([—{2}])([^—])', r"\1\n\2", sentences)  # 破折号分隔符
    sentences = re.sub('([-{2}])([^-])', r"\1\n\2", sentences)  # 破折号分隔符
    sentences = sentences.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences_list = sentences.split("\n")
    sentences_list_len = [len(sent) for sent in sentences_list]

    new_sentence_list = []
    idx = 0
    while idx < len(sentences_list_len):
        for jdx in range(idx + 1, len(sentences_list_len) + 1):
            if idx < jdx -1 and (sum(sentences_list_len[idx:jdx])) > max_content_length:
                new_sentence_list.append("".join(sentences_list[idx:jdx - 1]))
                idx = jdx - 1
                break
            elif idx == jdx -1 and (sum(sentences_list_len[idx:jdx])) > max_content_length:
                new_sentence_list.append("".join(sentences_list[idx:jdx]))
                idx = jdx
                break
            elif jdx == len(sentences_list_len):
                new_sentence_list.append("".join(sentences_list[idx:jdx]))
                idx = len(sentences_list_len)
                break

    return new_sentence_list

def cut_sentences(sentences, max_content_length = MAX_CONTENT_LENGTH):
    # Python中文文本分句source :https://www.pythonheidong.com/blog/article/53241/
    sentences = re.sub('([。！？\?])([^”’])', r"\1\n\2", sentences)  # 单字符断句符
    sentences = re.sub('(\.{6})([^”’])', r"\1\n\2", sentences)  # 英文省略号
    sentences = re.sub('(\…{2})([^”’])', r"\1\n\2", sentences)  # 中文省略号
    sentences = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sentences)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    sentences = sentences.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences_list = sentences.split("\n")
    sentences_list_len = [len(sent) for sent in sentences_list]

    new_sentence_list = []
    idx = 0
    while idx < len(sentences_list_len):
        for jdx in range(idx + 1, len(sentences_list_len) + 1):
            if idx < jdx -1 and (sum(sentences_list_len[idx:jdx])) > max_content_length:
                new_sentence_list.append("".join(sentences_list[idx:jdx - 1]))
                idx = jdx - 1
                break
            elif idx == jdx -1 and (sum(sentences_list_len[idx:jdx])) > max_content_length:
                #new_sentence_list.append("".join(sentences_list[idx:jdx]))
                new_sentence_list += cut_sub_sentences("".join(sentences_list[idx:jdx]))
                idx = jdx
                break
            elif jdx == len(sentences_list_len):
                new_sentence_list.append("".join(sentences_list[idx:jdx]))
                new_sentence_list.append("".join(sentences_list[idx:jdx]))
                idx = len(sentences_list_len)
                break

    return new_sentence_list


def run_proc(idx, n, file_list, input_folder, output_folder, entity_occurred_folder, display_wordseg = False):
    for i in range(len(file_list)):
        if i % n == idx:
            input_name = file_list[i]
            print("input_name=%s" % input_name)
            target = input_name.replace(input_folder, output_folder)
            entity_occurred_file = input_name.replace(input_folder, entity_occurred_folder)
            folder = '/'.join(target.split('/')[:-1])
            entity_occurred_foler = '/'.join(entity_occurred_file.split('/')[:-1])
            print ("target=%s" % target)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if not os.path.exists(entity_occurred_foler):
                os.makedirs(entity_occurred_foler)

            soup = BeautifulSoup(open(input_name), features="html5lib")
            #print ("soup=%s" % soup)
            docs = soup.find_all('doc', id = True, url = True, title = True)
            #print ("doc=%s" % docs)
            print ("len doc=%s" % str(len(docs)))

            fout = open(target, "w")
            entity_occurred_fout = open(entity_occurred_file, "w")

            for doc in docs:
                entity_occurred_list = []
                content = doc.get_text()
                print (content)

                while content[0] == "\n":
                    content = content[1:]
                content = [x.strip() for x in content.split("\n")]
                # begin with title: content[1:0]; begin without title: content[0:]
                #content = "".join(content[1:])
                # print (content)
                print ("len content=%s" % str(len(content)))
                #print (content)
                #exit(0)

                new_content_list = []
                word_segment_list = []
                # begin with title: content[1:0]; begin without title: content[0:]
                for content_ele in content[1:]:
                    print ("len_content_ele=%s" % str(len(content_ele)))
                    if content_ele == "":
                        continue
                    # control the length
                    if len(content_ele) > MAX_CONTENT_LENGTH:
                        content_ele_list = cut_sentences(content_ele, MAX_CONTENT_LENGTH)
                        print (content_ele_list)
                        # 执行错误，例如文本过长，或者其他输入格式上的错误，则抛弃该句子。
                        if len(content_ele_list) == 0:
                            continue
                    else:
                        content_ele_list = [content_ele]

                    for content_ele in content_ele_list:
                        result = entity_linking(content_ele)
                        word_segment = result["cuts"]
                        word_segment_list.extend(word_segment)

                        entities = result["entities"]
                        # char_segment = list(content_ele)

                        print ("word_segment=%s" % word_segment)
                        print ("entities=%s" % entities)
                        # print ("char_segment=%s" % char_segment)

                        new_content_ele = ""
                        last_end = 0
                        for [location, entity_name] in entities:
                            begin = location[0]
                            end = location[1]
                            if last_end <= begin:
                                new_content_ele += "%s<a href=%s>%s</a>" % \
                                              (content_ele[last_end : begin], \
                                               entity_name, \
                                               content_ele[begin: end])
                                print (new_content_ele)
                                last_end = end

                                entity_occurred_list.append(entity_name)

                        new_content_ele += content_ele[last_end:]

                        print (new_content_ele)
                        new_content_list.append(new_content_ele)
                print ("new_content_list=%s" % new_content_list)
                print ("entity_occurred_list=%s" % entity_occurred_list)

                #if doc.get_text(strip=True):
                id = doc['id']
                title = doc['title']
                url = doc['url']
                head = "<doc id=%s title=%s url=%s>" % (id, title, url)
                foot = "</doc>"

                word_seg_str = "[_seg_]".join(word_segment_list) if display_wordseg else ""
                display_info = "%s\n%s\n%s\n[_cutcutcut_]%s\n%s\n" % \
                               (head, title, "\n".join(new_content_list), word_seg_str, foot)

                fout.write(display_info)
                entity_occurred_fout.write("%s\n" % "\t".join(entity_occurred_list))
                time.sleep(1)

                #fout.close()
                #entity_occurred_fout.close()
                #exit(0)

            fout.close()
            entity_occurred_fout.close()


import sys

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print ("Usage: python create_entity_linking.py process_num input_folder output_folder entity_occurred_folder")
        exit(0)
        
    n = int(sys.argv[1])
    input_folder = sys.argv[2] + "/"
    output_folder = sys.argv[3] + "/"
    entity_occurred_folder = sys.argv[4] + "/"
    
    file_list = get_file_list(input_folder)
    
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i,n, file_list, input_folder, output_folder, entity_occurred_folder))
    p.close()
    p.join()