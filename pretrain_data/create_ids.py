import tokenization
import sys
import os
from multiprocessing import Pool
#from nltk.tokenize import sent_tokenize
import math

#vocab_file = "ernie_base/vocab.txt"
#entity_map_file = "alias_entity.txt"
#image_map_file = "alias_image.txt"
#input_folder = "../pretrain_data/ann"
#output_folder = "../pretrain_data/raw"
do_lower_case = True

# 用于分句子
# 版本为python3，如果为python2需要在字符串前面加上u
# 原文 https://blog.csdn.net/blmoistawinde/article/details/82379256
import re

def cut_sentence(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def get_tokenizer(vocab_file, do_lower_case = do_lower_case):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    return tokenizer

def get_file_list(input_folder):
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))

    part = int(math.ceil(len(file_list) / 20.))
    file_list = [file_list[i:i+part] for i in range(0, len(file_list), part)]
    return file_list

# load entity dict
def get_entity_dict(entity_map_file):
    d_ent = {}
    with open(entity_map_file, "r") as fin:
        for line in fin:
            line_list = line.strip().split("\t")
            if len(line_list) != 2:
                continue
            d_ent[line_list[0]] = line_list[1]
    return d_ent

# load image dict
def get_image_dict(image_map_file):
    d_image = {}
    with open(image_map_file, "r") as fin:
        for line in fin:
            line_list = line.strip().split("\t")
            if len(line_list) != 2:
                continue
            d_image[line_list[0]] = line_list[1]
    return d_image

def run_proc(idx, n, file_list, input_folder, output_folder, tokenizer, d_ent, d_image):
    for i in range(len(file_list)):
        if i % n == idx:
            target = "{}/{}".format(output_folder, i)
            fout_text = open(target+"_token", "w")
            fout_ent = open(target+"_entity", "w")
            fout_img = open(target+"_image", "w")
            #fout_raw_token = open(target+"_raw_token", "w")
            input_names = file_list[i]

            #fout_raw_token.write("[PAD]\n[unused0]\n[unused1]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
            sep_id = tokenizer.convert_tokens_to_ids(["sepsepsep"])[0]
            for input_name in input_names:
                print(input_name)
                fin = open(input_name, "r")

                for doc in fin:
                    print ("text=%s" % doc)
                    if doc.strip() == "":
                        continue
                    doc_text = doc.strip().split("[_cutcutcut_]")
                    doc = doc_text[0]
                    wordsegment = doc_text[1]

                    segs = doc.split("[_end_]")
                    content = segs[0]
                    #sentences = sent_tokenize(content)
                    sentences = cut_sentence(content)
                    print ("sentence=%s" % sentences)

                    ent_map_segs = segs[1:]
                    ent_maps = {}
                    img_maps = {}
                    for x in ent_map_segs:
                        v = x.split("[_map_]")
                        if len(v) != 2:
                            continue
                        if v[1] in d_ent:
                            ent_maps[v[0]] = d_ent[v[1]]
                        if v[1] in d_image:
                            img_maps[v[0]] = d_image[v[1]]
                    
                    text_out = [len(sentences)]
                    ent_out = [len(sentences)]
                    img_out = [len(sentences)]

                    print ("display len(sentences)=%s" % str(len(sentences)))
                    for sent in sentences:
                        print ("sentence begin...")
                        tokens = tokenizer.tokenize(sent)
                        anchor_ent_segs = [x.strip() for x in sent.split("sepsepsep")]
                        ent_result = []
                        img_result = []
                        for x in anchor_ent_segs:
                            if x in ent_maps and x in img_maps:
                                ent_result.append(ent_maps[x])
                                img_result.append(img_maps[x])
                            elif x in ent_maps:
                                ent_result.append(ent_maps[x])
                                img_result.append("#UNK#")
                            else:
                                ent_result.append("#UNK#")
                                img_result.append("#UNK#")

                        cur_seg = 0

                        new_text_out = []
                        new_ent_out = []
                        new_img_out = []

                        print("tokens=%s" % tokens)
                        #fout_raw_token.write("%s\n" % "\n".join(tokens))
                        for token in tokenizer.convert_tokens_to_ids(tokens):
                            if token != sep_id:
                                new_text_out.append(token)
                                new_ent_out.append(ent_result[cur_seg])
                                new_img_out.append(img_result[cur_seg])
                            else:
                                cur_seg += 1
                        print ("new_text_token=%s" % new_text_out)
                        print ("new_ent_out=%s" % new_ent_out)
                        print ("new_img_out=%s" % new_img_out)

                        if len(new_ent_out) != 0:
                            ent_out.append(len(new_ent_out))
                            ent_out.extend(new_ent_out)
                            img_out.append(len(new_img_out))
                            img_out.extend(new_img_out)
                            text_out.append(len(new_text_out))
                            text_out.extend(new_text_out)
                        else:
                            text_out[0] -= 1
                            ent_out[0] -= 1
                            img_out[0] -= 1
                        print ("text_out=%s" % text_out)
                        print ("ent_out=%s" % ent_out)
                        print ("img_out=%s" % img_out)
                    fout_ent.write("\t".join([str(x) for x in ent_out])+"\n")
                    fout_img.write("\t".join([str(x) for x in img_out])+"\n")
                    fout_text.write("\t".join([str(x) for x in text_out])+"\n")
                fin.close()
            fout_ent.close()
            fout_text.close()
            #fout_raw_token.close()

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print ("Usage: python3 create_ids.py process_num input_folder output_folder vocab_file entity_map_file image_map_file")
        exit(0)
        
    n = int(sys.argv[1])
    input_folder = sys.argv[2] + "/"
    output_folder = sys.argv[3] + "/"
    vocab_file = sys.argv[4]
    entity_map_file = sys.argv[5]
    image_map_file = sys.argv[6]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_list = get_file_list(input_folder)
    tokenizer = get_tokenizer(vocab_file)
    d_ent = get_entity_dict(entity_map_file)
    d_image = get_image_dict(image_map_file)
        
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i,n, file_list, input_folder, output_folder, tokenizer, d_ent, d_image))
    p.close()
    p.join()


