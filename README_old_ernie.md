## ERNIE

Source code and dataset for "[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)"

### Reqirements:

* Pytorch>=0.4.1
* Python3
* tqdm
* boto3
* requests
* apex (If you want to use fp16, you should make sure the commit is `880ab925bce9f817a93988b021e12db5f67f7787`.)

#### Prepare Pre-train Data

Run the following command to create training instances.

```shell
  # Download Wikidump
  wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
  # Download alise
  wget -c https://cloud.tsinghua.edu.cn/f/a519318708df4dc8a853/?dl=1 -O alias_entity.txt
  # WikiExtractor
  python3 pretrain_data/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o pretrain_data/output -l --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 4
  # Modify anchors with 4 processes
  python3 pretrain_data/extract.py 4
  # Preprocess with 4 processes
  python3 pretrain_data/create_ids.py 4
  # create instances
  python3 pretrain_data/create_insts.py 4
  # merge
  python3 code/merge.py
```

Run the following command to pretrain:

```
  python3 code/run_pretrain.py --do_train --data_dir pretrain_data/merge --bert_model ernie_base --output_dir pretrain_out/ --task_name pretrain --fp16 --max_seq_length 256
```

### Pre-trained Model

Download pre-trained knowledge embedding from [Google Drive](https://drive.google.com/open?id=14VNvGMtYWxuqT-PWDa8sD0e7hO486i8Y)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/ebcfdb8975b740d4b60a/) and extract it.

```shell
tar -xvzf kg_embed.tar.gz
```

Download pre-trained ERNIE from [Google Drive](https://drive.google.com/open?id=1DVGADbyEgjjpsUlmQaqN6i043SZvHPu5)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8df2a3a6261e4643a68f/) and extract it.

```shell
tar -xvzf ernie_base.tar.gz
```

Note that the extraction may be not completed in Windows.

### Fine-tune

As most datasets except FewRel don't have entity annotations, we use [TAGME](<https://tagme.d4science.org/tagme/>) to extract the entity mentions in the sentences and link them to their corresponding entitoes in KGs. We provide the annotated datasets [Google Drive](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/32668247e4fd4f9789f2/).

```shell
tar -xvzf data.tar.gz
```

In the root directory of the project, run the following codes to fine-tune ERNIE on different datasets.

**FewRel:**

```bash
python3 code/run_fewrel.py   --do_train   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --fp16   --loss_scale 128
# evaluate
python3 code/eval_fewrel.py   --do_eval   --do_lower_case   --data_dir data/fewrel/   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel   --fp16   --loss_scale 128
```

**TACRED:**

```bash
python3 code/run_tacred.py   --do_train   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --fp16   --loss_scale 128 --threshold 0.4
# evaluate
python3 code/eval_tacred.py   --do_eval   --do_lower_case   --data_dir data/tacred   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0   --output_dir output_tacred   --fp16   --loss_scale 128 --threshold 0.4
```

**FIGER:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/FIGER   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir output_figer  --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2
# evaluate
python3 code/eval_figer.py    --do_eval   --do_lower_case   --data_dir data/FIGER   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 2048   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir output_figer  --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2
```

**OpenEntity:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --fp16 --loss_scale 128
# evaluate
python3 code/eval_typing.py   --do_eval   --do_lower_case   --data_dir data/OpenEntity   --ernie_model ernie_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open --threshold 0.3 --fp16 --loss_scale 128
```

Some code is modified from the **pytorch-pretrained-BERT**. You can find the exlpanations of most parameters in [pytorch-pretrained-BERT](<https://github.com/huggingface/pytorch-pretrained-BERT>). 

As the annotations given by TAGME have confidence score, we use `--threshlod` to set the lowest confidence score and choose the annotations whose scores are higher than `--threshold`. In this experiment, the value is usually `0.3` or `0.4`.

The script for the evaluation of relation classification just gives the accuracy score. For the macro/micro metrics, you should use `code/score.py` which is from [tacred repo](<https://github.com/yuhaozhang/tacred-relation>).

```shell
python3 code/score.py gold_file pred_file
```

You can find `gold_file` and `pred_file` on each checkpoint in the output folder (`--output_dir`).

**New Tasks:**

If you want to use ERNIE in new tasks, you should follow these steps:

* Use an entity-linking tool like TAGME to extract the entities in the text
* Look for the Wikidata ID of the extracted entities
* Take the text and entities sequence as input data

Here is a quick-start example (`code/example.py`) using ERNIE for Masked Language Model. We show how to annotate the given sentence with TAGME and build the input data for ERNIE. Note that it will take some time (around 5 mins) to load the model.

```shell
# If you haven't installed tagme
pip install tagme
# Run example
python3 code/example.py
```

### Cite

If you use the code, please cite this paper:

```
@inproceedings{zhang2019ernie,
  title={{ERNIE}: Enhanced Language Representation with Informative Entities},
  author={Zhang, Zhengyan and Han, Xu and Liu, Zhiyuan and Jiang, Xin and Sun, Maosong and Liu, Qun},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```



