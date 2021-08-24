"""Code by Noa Garcia and Yuta Nakashima"""
import logging
import os
import pickle
import re

import argparse
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def save_obj(obj, filename, verbose=True):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    if verbose:
        logger.info("Saved object to %s." % filename)


def load_obj(filename, verbose=True):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    if verbose:
        logger.info("Load object from %s." % filename)
    return obj


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(df, out, label, index):
    qtypes = df['QType'].to_list()

    acc_total, acc_vis, acc_text, acc_tem, acc_know = 0, 0, 0, 0, 0
    num_vis, num_text, num_tem, num_know = 0, 0, 0, 0

    for o, l, i in zip(out, label, index):

        if o == l:
            acc_total += 1

        qtype = qtypes[i]

        if qtype == 'visual':
            num_vis += 1
            if o == l:
                acc_vis += 1
        elif qtype == 'textual':
            num_text += 1
            if o == l:
                acc_text += 1
        elif qtype == 'temporal':
            num_tem += 1
            if o == l:
                acc_tem += 1
        elif qtype == 'knowledge':
            num_know += 1
            if o == l:
                acc_know += 1

    acc_total = acc_total / len(out)
    acc_vis = acc_vis / num_vis
    acc_text = acc_text / num_text
    acc_tem = acc_tem / num_tem
    acc_know = acc_know / num_know

    logger.info('--- Accuracy')
    logger.info('Total: %.03f' % acc_total)
    logger.info('Visual : %.03f' % acc_vis)
    logger.info('Textual : %.03f' % acc_text)
    logger.info('Temporal : %.03f' % acc_tem)
    logger.info('Knowledge : %.03f' % acc_know)
    logger.info('------')

    return acc_total, acc_vis, acc_text, acc_tem, acc_know


def accuracy_val(out, label):
    acc_total = 0

    for o, l in zip(out, label):

        if o == l:
            acc_total += 1

    acc_total = acc_total / len(out)

    logger.info('--- Accuracy')
    logger.info('Total: %.03f' % acc_total)
    logger.info('------')

    return acc_total


"""Code by InterDigital"""
def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

SCENE_BASED_STREAMS = ["dialog", "video", "scene_dialog_summary"]
EPISODE_BASED_STREAMS = ["plot", "episode_dialog_summary"]

SCENE_SUMMARY_CSV = "scene_summary.csv"
EPISODE_SUMMARY_CSV = "episode_summary.csv"
TBBT_SUMMARIES_CSV = 'tbbt_summaries.csv'
SCENES_DESCRIPTIONS_CSV = 'scenes_descriptions.csv'
KNOWIT_DATA_TEST_CSV = 'knowit_data_test.csv'

def create_folder_with_timestamp(path, load_pretrained_model_exists):
    """
    Makes directory with timestamp suffix if a new directory needed, otherwise returns the given path
    :param path:
    :param load_pretrained_model_exists:
    :raise FileNotFoundError: If the given path is not exist when the pretrained model wanted to be used
    :return:
    """
    if not load_pretrained_model_exists:
        os.makedirs(path)
    elif not os.path.exists(path):
        raise FileNotFoundError

    return path


def create_folder(path):
    """
      Makes directory if not exist, returns the given path
      :param path
      :return: path
      """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def str2bool(v):
  if isinstance(v, bool):
      return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
  else:
      raise argparse.ArgumentTypeError('Boolean value expected.')


"""Code by Noa Garcia and Yuta Nakashima"""
def clean_html(raw_html):
    """
    Cleans html tags from :param raw_html
    :param raw_html:
    :return: cleaned text
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def truncate_seq_pair_inv(tokens_a, tokens_b, max_length):
    """
    Truncate pair of sequences if longer than max_length

    :param tokens_a:
    :param tokens_b:
    :param max_length:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def load_knowit_data(args, split_name):
    assert split_name in ["train", "val", "test"]
    input_file = os.path.join(args.data_dir, 'knowit_data_' + split_name + '.csv')
    df = pd.read_csv(input_file, delimiter='\t')
    logger.info('Loaded file %s.' % input_file)
    return df


