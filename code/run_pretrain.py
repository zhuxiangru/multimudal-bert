# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling import BertForPreTraining
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    
    vecs = []
    vecs.append([0]*200) # 扩充CLS的位置，其他所有索引向后+1.
    with open("config_data/kg_embed/entity2vec.vec", 'r') as fin:
    #with open("pretrain_data/config_data/entity2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            #vec = [float(x) for x in vec if x != ""]
            vec = [float(x) for x in vec]
            vecs.append(vec)
    print("vecs_len=%s" % str(len(vecs)))
    print("vecs_dim=%s" % str(len(vecs[0])))
    ent_embed = torch.FloatTensor(vecs)
    ent_embed = torch.nn.Embedding.from_pretrained(ent_embed)
    #ent_embed = torch.nn.Embedding(5041175, 100)

    logger.info("Shape of entity embedding: "+str(ent_embed.weight.size()))

    vecs = []
    vecs.append([0] * 4096)  # 扩充CLS的位置，其他所有索引向后+1.
    with open("config_data/kg_embed/image2vec.vec", 'r') as fin:
    #with open("pretrain_data/image_vec/image2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    print("vecs_len=%s" % str(len(vecs)))
    print("vecs_dim=%s" % str(len(vecs[0])))
    img_embed = torch.FloatTensor(vecs)
    img_embed = torch.nn.Embedding.from_pretrained(img_embed)

    logger.info("Shape of image embedding: " + str(img_embed.weight.size()))
    del vecs

    train_data = None
    num_train_steps = None
    if args.do_train:
        # TODO
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,BatchSampler
        import iterators
        #train_data = indexed_dataset.IndexedCachedDataset(args.data_dir)
        train_data = indexed_dataset.IndexedDataset(args.data_dir, fix_lua_indexing=True)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)

        def collate_fn(x):
            x = torch.LongTensor([xx for xx in x])

            entity_idx = x[:, 4 * args.max_seq_length:5 * args.max_seq_length]
            print("entity_idx=%s" % entity_idx)
            image_idx = x[:, 6 * args.max_seq_length:7 * args.max_seq_length]
            print("image_idx=%s" % image_idx)
            # Build candidate
            ent_uniq_idx = np.unique(entity_idx.numpy())
            print("ent_uniq_idx=%s" % str(ent_uniq_idx))
            img_uniq_idx = np.unique(image_idx.numpy())
            print("img_uniq_idx=%s" % str(img_uniq_idx))
            ent_candidate = ent_embed(torch.LongTensor(ent_uniq_idx + 1))
            ent_candidate = ent_candidate.repeat([n_gpu, 1])
            img_candidate = img_embed(torch.LongTensor(img_uniq_idx + 1))
            img_candidate = img_candidate.repeat([n_gpu, 1])
            # build entity labels
            ent_idx_dict = {}
            ent_idx_list = []
            for idx, idx_value in enumerate(ent_uniq_idx):
                ent_idx_dict[idx_value] = idx
                ent_idx_list.append(idx_value)
            ent_size = len(ent_uniq_idx)-1
            # build image labels
            img_idx_dict = {}
            img_idx_list = []
            for idx, idx_value in enumerate(img_uniq_idx):
                img_idx_dict[idx_value] = idx
                img_idx_list.append(idx_value)
            img_size = len(img_uniq_idx) - 1

            def ent_map(x):
                if x == -1:
                    return -1
                else:
                    rnd = random.uniform(0, 1)
                    if rnd < 0.05:
                        return ent_idx_list[random.randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x

            def img_map(x):
                if x == -1:
                    return -1
                else:
                    rnd = random.uniform(0, 1)
                    if rnd < 0.05:
                        return img_idx_list[random.randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x

            ent_labels = entity_idx.clone()
            ent_idx_dict[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: ent_idx_dict[x])

            entity_idx.apply_(ent_map)
            ent_emb = ent_embed(entity_idx+1)
            ent_mask = entity_idx.clone()
            ent_mask.apply_(lambda x: 0 if x == -1 else 1)
            ent_mask[:,0] = 1

            img_labels = image_idx.clone()
            img_idx_dict[-1] = -1
            img_labels = img_labels.apply_(lambda x: img_idx_dict[x])

            image_idx.apply_(img_map)
            img_emb = img_embed(image_idx + 1)
            img_mask = image_idx.clone()
            img_mask.apply_(lambda x: 0 if x == -1 else 1)
            img_mask[:, 0] = 1

            input_ids = x[:,:args.max_seq_length]
            input_mask = x[:,args.max_seq_length:2*args.max_seq_length]
            segment_ids = x[:,2*args.max_seq_length:3*args.max_seq_length]
            masked_lm_labels = x[:,3*args.max_seq_length:4*args.max_seq_length]
            next_sentence_label = x[:,8*args.max_seq_length:]
            return input_ids, input_mask, segment_ids, masked_lm_labels, ent_emb, ent_mask, img_emb, img_mask, next_sentence_label, ent_candidate, ent_labels, img_candidate, img_labels

        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    print ("len(train_data)=%s" % len(train_data))
    # Prepare model
    model, missing_keys = BertForPreTraining.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    print ("param_optimizer:")
    #for param in model.named_parameters():
    #    print(param[0])

    #no_linear = ['layer.2.output.dense_ent', 'layer.2.intermediate.dense_1', 'bert.encoder.layer.2.intermediate.dense_1_ent', 'layer.2.output.LayerNorm_ent']
    #no_linear = [x.replace('2', '11') for x in no_linear]
    no_linear = ['layer.11.output.dense_entity', 'layer.11.output.LayerNorm_entity', 'layer.11.output.dense_image', 'layer.11.output.LayerNorm_entity']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in no_linear)]
    print ("param_optimizer--no_linear")
    #for param in param_optimizer:
    #    print (param[0])

    #param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in missing_keys)]
    #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm_ent.bias', 'LayerNorm_ent.weight']
    #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm_ent.bias', 'LayerNorm_ent.weight']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm_token.bias', 'LayerNorm_token.weight', 'LayerNorm_entity.bias', 'LayerNorm_entity.weight', 'LayerNorm_image.bias', 'LayerNorm_image.weight']
    optimizer_grouped_parameters = [
        # weight decay to avoid overfitting 
        # source: https://blog.csdn.net/program_developer/article/details/80867468
        # source: https://blog.csdn.net/m0_37531129/article/details/101390592
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        # the decay of bias and normalization.weight has nothing to do with weight decay
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # optimizer_grouped_parameters_display is only used to debug
#    optimizer_grouped_parameters_display = [
#        {'params': [(n,p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#        {'params': [(n,p) for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#        ]
#    print ("optimizer_grouped_parameters_display-0:")
#    for param in optimizer_grouped_parameters_display[0]['params']:
#        print (param[0])
#
#    print ("optimizer_grouped_parameters_display-1:")
#    for param in optimizer_grouped_parameters_display[1]['params']:
#        print (param[0])

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            #from apex.optimizers import FP16_Optimizer
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        #optimizer = FusedAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      bias_correction=False,
        #                      max_grad_norm=1.0)
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        #logger.info(dir(optimizer))
        #op_path = os.path.join(args.bert_model, "pytorch_op.bin")
        #optimizer.load_state_dict(torch.load(op_path))

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        import datetime
        fout = open(os.path.join(args.output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                print ("step=%s" % str(step))
                print ("len(batch)=%s" % str(len(batch)))
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, input_img, img_mask, next_sentence_label, ent_candidate, ent_labels, img_candidate, img_labels = batch
                print ("\ninput_ids.size=%s" % str(input_ids.size()))
                print ("input_mask.size=%s" % str(input_mask.size()))
                print ("segment_ids.size=%s" % str(segment_ids.size()))
                print ("masked_lm_labels.size=%s" % str(masked_lm_labels.size()))
                print ("input_ent.size=%s" % str(input_ent.size()))
                print ("ent_mask.size=%s" % str(ent_mask.size()))
                print ("input_img.size=%s" % str(input_img.size()))
                print ("img_mask.size=%s" % str(img_mask.size()))
                print ("next_sentence_label.size=%s" % str(next_sentence_label.size()))
                print ("ent_candidate.size=%s" % str(ent_candidate.size()))
                print ("ent_labels.size=%s" % str(ent_labels.size()))
                print ("img_candidate.size=%s" % str(img_candidate.size()))
                print ("img_labels.size=%s" % str(img_labels.size()))

                if args.fp16:
                    loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels,
                                                input_ent.half(), ent_mask, input_img.half(), img_mask,
                                                next_sentence_label, ent_candidate.half(), ent_labels,
                                                img_candidate.half(), img_labels)
                else:
                    loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels,
                                                input_ent, ent_mask, input_img, img_mask,
                                                next_sentence_label, ent_candidate, ent_labels,
                                                img_candidate, img_labels)


                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    original_loss = original_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print("\nloss=%s\n" % str(loss))

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                fout.write("{} {}\n".format(loss.item()*args.gradient_accumulation_steps, original_loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    # source: https://blog.csdn.net/m0_37531129/article/details/101390592
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    #if global_step % 1000 == 0:
                    #    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    #    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                    #    torch.save(model_to_save.state_dict(), output_model_file)
        fout.close()

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Save the optimizer
    #output_optimizer_file = os.path.join(args.output_dir, "pytorch_op.bin")
    #torch.save(optimizer.state_dict(), output_optimizer_file)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    # model.to(device)

    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)

    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
    #             logits = model(input_ids, segment_ids, input_mask)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)

    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy

    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1

    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_examples

    #     result = {'eval_loss': eval_loss,
    #               'eval_accuracy': eval_accuracy,
    #               'global_step': global_step,
    #               'loss': tr_loss/nb_tr_steps}

    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()

