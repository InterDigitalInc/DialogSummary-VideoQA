"""Code by Noa Garcia and Yuta Nakashima"""
import logging
import math
import os
from abc import ABC

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from utils import SCENE_BASED_STREAMS, EPISODE_BASED_STREAMS, clean_html, truncate_seq_pair_inv, load_knowit_data, \
    SCENE_SUMMARY_CSV, EPISODE_SUMMARY_CSV, TBBT_SUMMARIES_CSV, SCENES_DESCRIPTIONS_CSV


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSample(object):

    def __init__(self, qid, question, answer1, answer2, answer3, answer4, subtitles, scene_description, knowledge,
                 label, summary):
        """

        :param qid:
        :param question:
        :param answer1:
        :param answer2:
        :param answer3:
        :param answer4:
        :param subtitles:
        :param scene_description:
        :param knowledge:
        :param label:
        :param summary:
        """
        self.qid = qid
        self.question = question
        self.subtitles = subtitles
        self.knowledge = knowledge
        self.label = label
        self.scene_description = scene_description
        self.answers = [
            answer1,
            answer2,
            answer3,
            answer4,
        ]
        self.summary = summary


"""Code by InterDigital"""
class DataloaderFactory:
    @staticmethod
    def build(args, split, tokenizer):
        stream_name = args.train_name
        if stream_name in SCENE_BASED_STREAMS:
            return SceneInputBasedStreamData(args, split, tokenizer)
        elif stream_name in EPISODE_BASED_STREAMS:
            return EpisodeInputBasedStreamData(args, split, tokenizer)
        else:
            raise NotImplementedError

"""Code by Noa Garcia and Yuta Nakashima"""
def get_qa_labels(df, index, row):
    question = row['question']
    answer1 = row['answer1']
    answer2 = row['answer2']
    answer3 = row['answer3']
    answer4 = row['answer4']
    label = int(df['idxCorrect'].iloc[index] - 1)
    return answer1, answer2, answer3, answer4, label, question


class Dataloader(data.Dataset, ABC):
    def __init__(self, args, split, tokenizer):
        self.df = load_knowit_data(args, split)
        self.tokenizer = tokenizer
        self.split = split
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.samples = self.get_data(self.df)
        self.num_samples = len(self.samples)

    def get_data(self, df):
        raise NotImplementedError

    def __len__(self):
        return self.num_samples

"""Code by InterDigital"""
class EpisodeInputBasedStreamData(Dataloader):
    def __init__(self, args, split, tokenizer):
        if args.train_name == "plot":
            dfkg = pd.read_csv(os.path.join(args.data_dir, TBBT_SUMMARIES_CSV))
            self.recap_dict = dfkg.set_index('Episode').T.to_dict('list')
        elif args.train_name == "episode_dialog_summary":
            episode_summary_df = pd.read_csv(os.path.join(args.data_dir, EPISODE_SUMMARY_CSV),sep='\t')
            self.episode_summary_dict = episode_summary_df.set_index("episode_name").episode_summary.to_dict()
        else:
            raise NotImplementedError

        super().__init__(args, split, tokenizer)
        self.num_max_slices = args.num_max_slices
        self.stride = args.seq_stride

        logger.info('Data loader ready with {:d} samples'.format(self.num_samples))

    """Code by Noa Garcia and Yuta Nakashima"""
    def get_data(self, df):
        samples = []
        for index, row in df.iterrows():
            answer1, answer2, answer3, answer4, label, question = get_qa_labels(df, index, row)
            """Code by InterDigital"""
            if self.args.train_name == "episode_dialog_summary":
                episode = row.scene[:6]
                plot_summary = self.episode_summary_dict[episode]

            elif self.args.train_name == "plot":
                episode = row.scene[:6]
                season = episode[1:3]
                number = episode[4:6]
                idepi = int(str(int(season)) + number)
                plot_summary = self.recap_dict[idepi][0]
            else:
                raise NotImplementedError
            """Code by Noa Garcia and Yuta Nakashima"""
            samples.append(DataSample(qid=index, question=question, answer1=answer1, answer2=answer2, answer3=answer3,
                                      answer4=answer4, subtitles=None, scene_description=None, knowledge=plot_summary,
                                      label=label,
                                      summary=None))
        return samples

    def __getitem__(self, index):
        """
        Convert each sample into 4*num_max_slices BERT input sequences as:

        [CLS] + kg_part_1 + question + [SEP] + answer1 + [SEP]
        [CLS] + kg_part_1 + question + [SEP] + answer2 + [SEP]
        [CLS] + kg_part_1 + question + [SEP] + answer3 + [SEP]
        [CLS] + kg_part_1 + question + [SEP] + answer4 + [SEP]

        [CLS] + kg_part_2 + question + [SEP] + answer1 + [SEP]
        [CLS] + kg_part_2 + question + [SEP] + answer2 + [SEP]
        .
        .
        .
        [CLS] + kg_part_num_max_slices + question + [SEP] + answer4 + [SEP]

        sample = self.samples[index]
        :param index:
        """
        sample = self.samples[index]
        question_tokens = self.tokenizer.tokenize(sample.question)
        all_knowledge_tokens = self.tokenizer.tokenize(sample.knowledge)
        list_answer_tokens = []
        for answer in sample.answers:
            answer_tokens = self.tokenizer.tokenize(answer)
            list_answer_tokens.append(answer_tokens)

        # Compute maximum window length for knowledge slices based on question and answer lengths
        max_qa_len = len(question_tokens) + max([len(a) for a in list_answer_tokens])
        len_extra_tokens = 3
        len_kg_window = self.max_seq_length - max_qa_len - len_extra_tokens

        # Slice knowledge according to window and stride
        list_knowledge_tokens = []

        num_kg_pieces = min(math.ceil((len(all_knowledge_tokens) - len_kg_window) / self.stride) + 1,
                            self.num_max_slices)
        num_kg_pieces = max(num_kg_pieces, 1)
        for n in list(range(num_kg_pieces)):
            maxpos = min(len_kg_window + (self.stride * n), len(all_knowledge_tokens))
            tokens = all_knowledge_tokens[self.stride * n:maxpos]
            list_knowledge_tokens.append(tokens)

        # Transformer input features
        sample_input_ids = np.zeros((self.num_max_slices, len(sample.answers), self.max_seq_length))
        sample_input_mask = np.zeros((self.num_max_slices, len(sample.answers), self.max_seq_length))
        sample_segment_ids = np.zeros((self.num_max_slices, len(sample.answers), self.max_seq_length))
        for kg_index, knowledge_tokens in enumerate(list_knowledge_tokens):
            for answer_index, answer_tokens in enumerate(list_answer_tokens):
                """Code by InterDigital"""
                start_tokens = knowledge_tokens[:] + question_tokens[:]
                ending_tokens = answer_tokens

                """Code by Noa Garcia and Yuta Nakashima"""
                sequence_tokens = [self.tokenizer.cls_token] + start_tokens + [
                    self.tokenizer.sep_token] + ending_tokens + [self.tokenizer.sep_token]
                segment_ids = [0] * (len(start_tokens) + 2) + [1] * (len(ending_tokens) + 1)
                input_ids = self.tokenizer.convert_tokens_to_ids(sequence_tokens)
                input_mask = [1] * len(input_ids)

                padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                sample_input_ids[kg_index, answer_index, :] = input_ids
                sample_input_mask[kg_index, answer_index, :] = input_mask
                sample_segment_ids[kg_index, answer_index, :] = segment_ids

        sample_input_ids = torch.tensor(sample_input_ids, dtype=torch.long)
        sample_input_mask = torch.tensor(sample_input_mask, dtype=torch.long)
        sample_segment_ids = torch.tensor(sample_segment_ids, dtype=torch.long)
        qid = torch.tensor(sample.qid, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.long)
        return sample_input_ids, sample_input_mask, sample_segment_ids, qid, label

"""Code by InterDigital"""
class SceneInputBasedStreamData(Dataloader):
    def __init__(self, args, split, tokenizer):
        super().__init__(args, split, tokenizer)
        self.num_samples = len(self.samples)
        logger.info('Data loader ready with {:d} samples'.format(self.num_samples))

    def get_data(self, df):
        """
        Load data into list of DataSamples
        :param df:
        :return:
        """
        samples = []

        if self.args.train_name == "video":
            df_descriptions = pd.read_csv(os.path.join(self.args.data_dir, SCENES_DESCRIPTIONS_CSV),
                                          delimiter='\t')
            df_descriptions.replace(np.nan, '', inplace=True)
        elif self.args.train_name == "scene_dialog_summary":
            df_summaries = pd.read_csv(os.path.join(self.args.data_dir, SCENE_SUMMARY_CSV), sep="\t")

        """Code by Noa Garcia and Yuta Nakashima"""
        for index, row in df.iterrows():
            summary = None
            subtitles = None
            scene_description = None
            answer1, answer2, answer3, answer4, label, question = get_qa_labels(df, index, row)

            """Code by InterDigital"""
            if self.args.train_name == "dialog":
                subtitles = clean_html(row['subtitle'].replace('<br />', ' ').replace(' - ', ' '))
            elif self.args.train_name == "scene_dialog_summary":
                scene_name = row['scene']
                summary = df_summaries[df_summaries.scene == scene_name].summary.values[0]
            elif self.args.train_name == "video":
                scene_name = row['scene']
                scene_description = ''
                if len(df_descriptions[df_descriptions['Scene'] == scene_name]['Description']) > 0:
                    scene_description = df_descriptions[df_descriptions['Scene'] == scene_name]['Description'].values[0]
            else:
                raise NotImplementedError

            """Code by Noa Garcia and Yuta Nakashima"""
            samples.append(DataSample(qid=index, question=question, answer1=answer1, answer2=answer2, answer3=answer3,
                                      answer4=answer4, subtitles=subtitles, scene_description=scene_description,
                                      knowledge=None,
                                      label=label, summary=summary))
        return samples

    def __getitem__(self, index):
        """
        Convert each sample into 4 BERT input sequences as:
        [CLS] + subtitles + question + [SEP] + answer1 + [SEP]
        [CLS] + subtitles + question + [SEP] + answer2 + [SEP]
        [CLS] + subtitles + question + [SEP] + answer3 + [SEP]
        [CLS] + subtitles + question + [SEP] + answer4 + [SEP]
        :param index:
        :return:
        """

        sample = self.samples[index]

        """Code by InterDigital"""
        train_name = self.args.train_name
        if train_name == "dialog":
            text_tokens = self.tokenizer.tokenize(sample.subtitles)
        elif train_name == "scene_dialog_summary":
            text_tokens = self.tokenizer.tokenize(sample.summary)
        elif train_name == "video":
            text_tokens = self.tokenizer.tokenize(sample.scene_description)
        else:
            raise NotImplementedError

        """Code by Noa Garcia and Yuta Nakashima"""
        question_tokens = self.tokenizer.tokenize(sample.question)
        choice_features = []
        for answer_index, answer in enumerate(sample.answers):
            start_tokens = text_tokens[:] + question_tokens[:]
            ending_tokens = self.tokenizer.tokenize(answer)
            truncate_seq_pair_inv(start_tokens, ending_tokens, self.max_seq_length - 3)
            tokens = [self.tokenizer.cls_token] + start_tokens + [self.tokenizer.sep_token] + ending_tokens + [
                self.tokenizer.sep_token]
            segment_ids = [0] * (len(start_tokens) + 2) + [1] * (len(ending_tokens) + 1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            choice_features.append((tokens, input_ids, input_mask, segment_ids))

        input_ids = torch.tensor([data[1] for data in choice_features], dtype=torch.long)
        input_mask = torch.tensor([data[2] for data in choice_features], dtype=torch.long)
        segment_ids = torch.tensor([data[3] for data in choice_features], dtype=torch.long)
        qid = torch.tensor(sample.qid, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.long)
        return input_ids, input_mask, segment_ids, qid, label
