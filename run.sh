flag_entity_linking=0
flag_normalization=0
flag_numerical_token=0
flag_train=1
project_dir="/mnt/f/PyCharmProgram/ERNIE_zh_multimodal"
input_data="pretrain_data/input_data"
output_data="pretrain_data/output_data"
config_data="config_data"

workon venv_bert
#source ~/env/py36env/bin/activate

cd ${project_dir}

if [ ${flag_entity_linking} -eq 1 ]
then
  # 抽取wiki数据
  ls pretrain_data/mid_result/
  if [ ${?} -ne 0 ] ;then
    mkdir pretrain_data/mid_result/
  fi

  if false; then
    wiki_extract_input="/mnt/f/ProgramProject/zhwiki_data/zhwiki-latest-pages-articles1.xml-p1p162886"
    wiki_extract_output="pretrain_data/mid_result/extract/"
    python3 pretrain_data/WikiExtractor.py ${wiki_extract_input} --output ${wiki_extract_output} --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 8
  fi

  # 繁简体转化
  if false; then
    opencc_input="pretrain_data/mid_result/extract/"
    opencc_output="pretrain_data/mid_result/opencc/"
    ln -s ${opencc_input} ${input_data}

    which opencc
    if [ ${?} -ne 0 ] ;then
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
        opencc -c t2s -i ${inputfilepath} -o ${outputfilepath}
      done
    done
  fi
  echo "finish opencc t2s transform"

  # 实体链接
  #pip3 install html5lib   # 非常重要
  #pip3 install bs4
  entity_linking_input="pretrain_data/mid_result/opencc/"
  entity_linking_output="pretrain_data/mid_result/output/"
  entity_linking_mid_result="pretrain_data/mid_result/entity_occurred/"
  rm -rf ${entity_linking_output}/*
  python3 pretrain_data/create_entity_linking.py 1 ${entity_linking_input} ${entity_linking_output} ${entity_linking_mid_result}
  echo "finish create_entity_linking"

  # 统计得到entity的列表, 在测试小样本时用。大样本时，直接统计cn-dbpedia有多少实体即可
  alias_entity_input="pretrain_data/mid_result/entity_occurred/"
  alias_entity_output="pretrain_data/mid_result/entity_alias/"
  python3 pretrain_data/statistic_alias_entity.py 8 ${alias_entity_input} ${alias_entity_output}
  echo "finish statistic_alias_entity"

  # 去重、去空放到alias_entity.txt中
  # cat ${alias_entity_output}/*/* |sort -u|awk -F'\t' '{if($1!="") print $1"\tQ"NR}' > alias_entity.txt
  # cat ${alias_entity_output}/*/* |sort -u|awk -F'\t' '{print $1"\tQ"NR}' > tmp_alias_entity.txt
  cat ${alias_entity_output}/*/* |sort -u > tmp_alias_entity.txt
  # 规范化表示文本:xxxxxxxxxx sepsepsep entity text sepsepsep xxxxxxxxxx[_end_]entity text[_map_]entity[_end_]
  # 可能后面还有中文分词的结果[_cutcutcut_]word[_seg_]word[_seg_]......

  extract_entity_input="pretrain_data/mid_result/output/"
  extract_entity_output="pretrain_data/mid_result/ann/"
  python3 pretrain_data/extract.py 8 ${extract_entity_input} ${extract_entity_output}
  echo "finish extract_entity"
  cd ..
fi

if [ ${flag_normalization} -eq 1 ]
then

  ls config_data
  if [ ${?} -ne 0 ] ;then
    mkdir config_data
  fi

  ls config_data/kg_embed
  if [ ${?} -ne 0 ] ;then
    mkdir config_data/kg_embed
  fi

  mv tmp_alias_entity.txt config_data/
  #该部分的正确流程是通过OpenKE训练得到entity2id.txt和entity2vec.vec。之后所有实体链接到的entity都可以在entity2id.txt找到并找到应对vec
  #但是目前，全部的entity训练vec跑不起来（目前显存4G），目前只用训练文本中涉及到的entityi（4780个）与相关entity（215w+）训练得到vec。导致这部分处理麻烦。

  # 收集entity id
  # 如果没有，可以先通过alias_entity.txt涉及到的entity, 在OpenKE训练得到entity的embedding，即OpenKE/benchmark/cndbpedia_bert_without_attributes/的结果
  ls config_data/alias_entity_name2uri.txt
  if [ ${?} -ne 0 ] ;then
    ls config_data/kg_embed/entity2id.txt
    if [ ${?} -ne 0 ] ;then
      scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/entity2id.txt config_data/kg_embed/
      echo "finish download entity2id.txt"
    fi
    python3 pretrain_data/aliagn_trained_vecs.py config_data/kg_embed/entity2id.txt config_data/tmp_alias_entity.txt config_data/alias_entity_name2uri.txt config_data/alias_entity_uri2id.txt entity
    echo "finish generate alias_entity_name2uri.txt, alias_entity_uri2id.txt"
  fi

  # 收集entity embedding
  ls config_data/entity2vec.vec
  if [ ${?} -ne 0 ] ;then
    scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/entity2vec.vec config_data/kg_embed/
  fi

  # 如果没有，可以先随便生成一个
  #cd pre-trained
  #entity_num=`cat alias_entity.txt |wc -l`
  #echo ${entity_num} > kg_embed/entity2id.txt
  #cat alias_entity.txt |awk -F'\t' '{print $2"\t"NR}' >> kg_embed/entity2id.txt
  #cd ..
  #cat pretrain_data/kg_embed/entity2vec.vec |head -${entity_num} > kg_embed/entity2vec.vec

  # 收集image id
  ls config_data/alias_image_name2uri.txt
  if [ ${?} -ne 0 ] ;then
    ls config_data/kg_embed/image2id.txt
    if [ ${?} -ne 0 ] ;then
      scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/image2id.txt config_data/kg_embed/
      echo "finish download image2id.txt"
    fi
    # 注意，这里第三个参数是tmp_alias_entity.txt，不是tmp_alias_image.txt，因为这里只存放候选的实体，具体有没有图片要根据程序中筛选
    python3 pretrain_data/aliagn_trained_vecs.py config_data/kg_embed/image2id.txt config_data/tmp_alias_entity.txt config_data/alias_image_name2uri.txt config_data/alias_image_uri2id.txt image
    echo "finish generate alias_image_name2uri.txt, alias_image_uri2id.txt"
  fi
  # 收集image vector
  # 如果没有，可以先随便生成一个，例如随机数
  ls config_data/entity2vec.vec
  if [ ${?} -ne 0 ] ;then
    scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/image2vec.vec config_data/kg_embed/
  fi

  ls config_data/vocab.txt
  if [ ${?} -ne 0 ] ;then
    # 中文字符集vocab.txt的构造方法（字/词）嘛……暂略；直接下载字级别的vocab.txt
    scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/vocab.txt config_data/
    # password: zhuxiangru
  fi
  # nltk.tokenize 的sent_tokenize 会load包punkt，所以下载. import nltk;nltk.download('punkt') 下载后，存放在 /home/zhuxiangru/nltk_data

  # 符号化表示token
  cread_ids_input="pretrain_data/mid_result/ann/"
  cread_ids_output="pretrain_data/mid_result/raw/"
  vocab_file="config_data/vocab.txt"
  entity_map_file="config_data/alias_entity_name2uri.txt"
  image_map_file="config_data/alias_image_name2uri.txt"
  rm -rf ${cread_ids_output}/*
  python3 pretrain_data/create_ids.py 8 ${cread_ids_input} ${cread_ids_output} ${vocab_file} ${entity_map_file} ${image_map_file}
fi

if [ ${flag_numerical_token} -eq 1 ]
then

  # 数值化表示token,并写入bin文件
  create_insts_input="pretrain_data/mid_result/raw/"
  create_insts_output="pretrain_data/mid_result/data/"
  create_insts_run_python_file="pretrain_data/create_instances.py"
  vocab_file="config_data/vocab.txt"
  entity2id_file="config_data/alias_entity_uri2id.txt"
  image2id_file="config_data/alias_image_uri2id.txt"
  python3 pretrain_data/create_insts.py 8 ${create_insts_input} ${create_insts_output} ${create_insts_run_python_file} ${vocab_file} ${entity2id_file} ${image2id_file}
  # 根据生成的索引，Qn---n-1对应的是第n-1个vec。-1对应空vec。为了统一表示，预训练时会统一编号+1，在最前面增加-1对应全0向量，其他向量依次后推，Q2071----2071对应索引为2072的向量（索引本应该是2071）

  # merge
  merge_input="pretrain_data/mid_result/data"
  merge_output="pretrain_data/mid_result/merge"
  merge_output_bin_file="merge.bin"
  merge_output_idx_file="merge.idx"
  python3 pretrain_data/merge.py ${merge_input} ${merge_output} ${merge_output_bin_file} ${merge_output_idx_file}
  ln -s ${merge_output} ${output_data}
fi

if [ ${flag_train} -eq 1 ]
then
  # 下载ernie_base文件夹（pre-trained ERNIE）https://cloud.tsinghua.edu.cn/f/8df2a3a6261e4643a68f/, pytroch.bin可以没有，但是config.json文件需要有。是bert的配置文件。
  ls config_data/ernie_base
  if [ ${?} -ne 0 ] ;then
    mkdir config_data/ernie_base
  fi
  ls config_data/ernie_base/config.json
  if [ ${?} -ne 0 ] ;then
    scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/config.json config_data/ernie_base/
    scp zhuxiangru@10.131.246.48:/home/zhuxiangru/share/pytorch_model.bin config_data/ernie_base/
    # password: zhuxiangru
  fi
  rm -rf pretrain_out
  mkdir pretrain_out
  #python3 code/run_pretrain.py --do_train --data_dir pretrain_data/merge --bert_model pretrain_data/ernie_base --output_dir pretrain_out/ --task_name pretrain --fp16 --max_seq_length 256
  python3 code/run_pretrain.py --do_train --data_dir ${output_data}/merge --bert_model config_data/ernie_base/ --output_dir pretrain_out/ --task_name pretrain --fp16 --max_seq_length 256
fi


