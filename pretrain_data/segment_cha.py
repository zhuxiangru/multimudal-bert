# encoding = utf-8
import sys
import re

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars

if __name__ == '__main__':
    char_dic = {}
    for line in sys.stdin:
        char_list = seg_char(line.strip())
        for ele in char_list:
            if ele not in char_dic:
                char_dic[ele] = None

    for key in char_dic:
        print (key)