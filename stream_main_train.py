# coding=utf-8
"""Code by Noa Garcia and Yuta Nakashima"""
import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from torch import nn

from stream_data_sample import DataloaderFactory
from train_stream import stream_training, stream_embeddings
from utils import EPISODE_BASED_STREAMS, create_folder, str2bool

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True, type=bool)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--patience", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--seq_stride", default=100, type=int)
    parser.add_argument("--num_max_slices", default=10, type=int)
    parser.add_argument("--train_name", type=str, required=True, help="dialog, video, summary, episode_summary, plot")

    """Code by InterDigital"""
    parser.add_argument("--mini_batch_size", default=None, type=int)
    parser.add_argument("--temporal_attention_temperature", default=2, type=float)
    parser.add_argument("--temporal_attention",default=True, type=str2bool)
    parser.add_argument("--stream_train_folder_path", default='Training/', type=str)

    args, unknown = parser.parse_known_args()
    return args

"""Code by Noa Garcia and Yuta Nakashima"""
class StreamTransformer(BertPreTrainedModel):

    def __init__(self, config):
        super(StreamTransformer, self).__init__(config)
        self.args = args
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        if self.args.train_name in EPISODE_BASED_STREAMS:
            self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        if self.args.train_name in EPISODE_BASED_STREAMS:
            num_choices = input_ids.shape[2]
            num_slices = input_ids.shape[1]
        else:
            num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.args.train_name in EPISODE_BASED_STREAMS:
            unpooled_reshaped_logits = logits.view(-1, num_slices, num_choices)

            """Code by InterDigital"""
            if self.args.temporal_attention:
                # temporal attention
                a = torch.max(unpooled_reshaped_logits, dim=2)[0].unsqueeze(-1)
                s = nn.Softmax(dim=1)(a / self.args.temporal_attention_temperature)
                reshaped_logits = torch.matmul(s.transpose(1, 2), unpooled_reshaped_logits).squeeze(1)

            else:
                """Code by Noa Garcia and Yuta Nakashima"""
                reshaped_logits = torch.max(unpooled_reshaped_logits, dim=1)[0]

            pooled_output_slices = pooled_output.view(-1, num_slices, self.hidden_size)
            outputs = (reshaped_logits,) + (pooled_output_slices,) + (unpooled_reshaped_logits,)

        else:
            reshaped_logits = logits.view(-1, num_choices)
            outputs = (reshaped_logits,) + outputs[1:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


def pretrain_stream(args):
    # Create training and data directories
    base_model_path = os.path.join(args.stream_train_folder_path, args.train_name)
    base_embedding_path = os.path.join(base_model_path, 'embeddings')

    modeldir = create_folder(base_model_path)
    outdatadir = create_folder(base_embedding_path)

    with open(os.path.join(modeldir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    # Prepare GPUs
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Do training if there is not already a model in modeldir
    if not os.path.isfile(os.path.join(modeldir, 'pytorch_model.bin')):

        # Prepare model
        model = StreamTransformer.from_pretrained(args.bert_model, cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                                                                          'distributed_{}'.format(-1)))


        model.to(args.device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Load training data
        trainDataObject = DataloaderFactory.build(args, split='train', tokenizer=tokenizer)
        valDataObject = DataloaderFactory.build(args, split='val', tokenizer=tokenizer)

        # Start training
        logger.info('*** %s stream training ***' % args.train_name)
        stream_training(args, model, modeldir, n_gpu, trainDataObject, valDataObject)

    # For extracting stream embeddings, load trained weights
    model = StreamTransformer.from_pretrained(modeldir)
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Get stream embeddings for each dataset split
    logger.info('*** Get %s stream embeddings for each data split ***' % args.train_name)

    """Code by InterDigital"""
    for split in ["train", "val", "test"]:
        data_object = DataloaderFactory.build(args, split=split, tokenizer=tokenizer)
        stream_embeddings(args, model, outdatadir, data_object, split=split)
    logger.info('*** Pretraining %s stream done!' % args.train_name)

"""Code by Noa Garcia and Yuta Nakashima"""
if __name__ == "__main__":
    global args
    args = get_params()

    """Code by InterDigital"""
    logger.info("Arguments: %s" % json.JSONEncoder().encode(vars(args)))

    """Code by Noa Garcia and Yuta Nakashima"""
    pretrain_stream(args)
