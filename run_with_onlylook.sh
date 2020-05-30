flag_entity_linking=1
flag_normalization=0
flag_numerical_token=0
flag_train=0

cd "/mnt/f/PyCharmProgram/ERNIE_zh_multimodal"

if [ ${flag_entity_linking} -eq 1 ]
then
  # 抽取wiki数据
  python3 pretrain_data/WikiExtractor.py /mnt/f/ProgramProject/zhwiki_data/zhwiki-latest-pages-articles1.xml-p1p162886 --output pretrain_data/extract --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 8

  # 繁简体转化
  opencc_input="pretrain_data/extract/"
  opencc_output="pretrain_data/opencc/"
  which opencc
  if [ ${?} -nq 0 ] ;then
    sudo apt-get install opencc
  fi
  for dirname in `ls ${opencc_input}`
  do
    echo ${dirname}
    for filename in `ls ${opencc_input}"/"${dirname}`
    do
      echo ${dirname}"/"${filename}
      inputfilepath=${opencc_input}"/"${dirname}"/"${filename}
      outputfilepath=${opencc_output}"/"${dirname}"/"${filename}
      echo "inputpath:"${inputfilepath}
      echo "outputpath:"${outputfilepath}
          if [ ! -d ${opencc_output}"/"${dirname} ]
          then
              mkdir ${opencc_output}"/"${dirname}
          fi
      opencc -c -t2s -i ${inputfilepath} -o ${outputfilepath}
    done
  done

  # 实体链接
  pip install html5lib   # 非常重要
  rm -rf pretrain_data/output/*
  python3 pretrain_data/create_entity_linking.py 8

  # 统计得到entity的列表, 在测试小样本时用。大样本时，直接统计cn-dbpedia有多少实体即可
  python3 pretrain_data/statistic_alias_entity.py 8
  # 去重、去空放到alias_entity.txt中
  cat pretrain_data/entity_alias/*/wiki* |sort -u|awk -F'\t' '{if($1!="") print $1"\tQ"NR}' > alias_entity.txt
  # 规范化表示文本:xxxxxxxxxx sepsepsep entity text sepsepsep xxxxxxxxxx[_end_]entity text[_map_]entity[_end_]
  # 可能后面还有中文分词的结果[_cutcutcut_]word[_seg_]word[_seg_]......
  python3 extract.py 8
fi

if [ ${flag_normalization} -eq 1 ]
then
  ls pretrain_data/alias_entity.txt
  if [ ${?} -nq 0 ] ;then
    wget -c https://cloud.tsinghua.edu.cn/f/a519318708df4dc8a853/?dl=1 -O pretrain_data/alias_entity.txt
  fi
  # 下载ernie_base文件夹（pre-trained ERNIE）https://cloud.tsinghua.edu.cn/f/8df2a3a6261e4643a68f/
  # nltk.tokenize 的sent_tokenize 会load包punkt，所以下载. import nltk;nltk.download('punkt') 下载后，存放在 /home/zhuxiangru/nltk_data

  # 中文字符集vocab.txt的构造方法嘛……暂略

  # 符号化表示token
  rm pretrain_data/raw/*
  python3 pretrain_data/create_ids.py 8
fi

if [ ${flag_numerical_token} -eq 1 ]
then
  # 收集entity id
  cd pre-trained
  entity_num=`cat alias_entity.txt |wc -l`
  echo ${entity_num} > kg_embed/entity2id.txt
  cat alias_entity.txt |awk -F'\t' '{print $2"\t"NR}' >> kg_embed/entity2id.txt
  cd ..
  # 收集entity embedding,。。。。。没有。。。
  # 瞎编一个
  cat pretrain_data/kg_embed/entity2vec.vec |head -${entity_num} > kg_embed/entity2vec.vec

  # 数值化表示token
  python3 pretrain_data/create_insts.py 8

  # merge
  python3 code/merge.py
fi

if [ ${flag_train} -eq 1 ]
then
  python3 code/run_pretrain.py --do_train --data_dir pretrain_data/merge --bert_model pretrain_data/ernie_base --output_dir pretrain_out/ --task_name pretrain --fp16 --max_seq_length 256
fi

