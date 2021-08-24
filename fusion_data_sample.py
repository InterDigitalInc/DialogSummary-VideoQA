"""Code by Noa Garcia and Yuta Nakashima"""
import logging
import os

import numpy as np
from torch.utils import data

import utils
from utils import SCENE_BASED_STREAMS, EPISODE_BASED_STREAMS
from utils import load_knowit_data
from scipy.special import softmax


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionDataSample(data.Dataset):

    def __init__(self, args, split):
        df = load_knowit_data(args, split)
        self.labels = (df['idxCorrect'] - 1).to_list()
        """Code by InterDigital"""
        self.scene_based_features = []
        self.episode_based_features = []
        self.episode_logits_slices = []
        self.scene_based_stream_names = []
        self.episode_based_stream_names = []
        """Code by Noa Garcia and Yuta Nakashima"""
        self.args = args

        for stream in args.fuse_stream_list:

            base_embedding_path = os.path.join(args.stream_train_folder_path, stream, 'embeddings')
            embeddings = utils.load_obj(
                os.path.join(base_embedding_path, stream+'_stream_embeddings_%s.pckl' % split))

            if stream in SCENE_BASED_STREAMS:
                self.scene_based_stream_names.append(stream)
                self.scene_based_features.append(np.reshape(embeddings, (int(embeddings.shape[0] / 4), 4, 768)))
            elif stream in EPISODE_BASED_STREAMS:
                self.episode_based_stream_names.append(stream)
                episode_based_reshaped_feature = np.reshape(embeddings[0], (
                int(embeddings[0].shape[0] / 4), args.__dict__['num_max_slices_' + stream], 4, 768))
                self.episode_based_features.append(episode_based_reshaped_feature)
                self.episode_logits_slices.append(embeddings[1])
            else:
                raise NotImplementedError

        self.num_samples = len(self.labels)
        logger.info('Dataloader with %d samples' % self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        label = self.labels[index]
        outputs = [label, index]
        inputs = []
        """Code by InterDigital"""
        scene_based_inputs = []
        episode_based_inputs = []
        for stream in self.scene_based_features:
            scene_based_inputs.append(stream[index, :])
        for stream, slice in zip(self.episode_based_features, self.episode_logits_slices):
            stream_slices = stream[index, :]
            stream_logits_slice = slice[index, :]

            if self.args.part_selection_with_soft_temporal_attention:
                a = np.max(stream_logits_slice, axis=1).reshape(1, -1)
                s = softmax(a / self.args.ss_max_temperature, axis=1)
                results_embeddings = np.matmul(s, stream_slices.reshape(s.shape[1], -1)).reshape(4, 768)
                episode_based_inputs.append(results_embeddings)
            else:
                idx_slice, _ = np.unravel_index(stream_logits_slice.argmax(), stream_logits_slice.shape)
                episode_based_inputs.append(stream_slices[idx_slice, :])

        for stream in self.args.fuse_stream_list:
            if stream in self.scene_based_stream_names:
                stream_names_index = self.scene_based_stream_names.index(stream)
                inputs.append(scene_based_inputs[stream_names_index])
            else:
                stream_names_index = self.episode_based_stream_names.index(stream)
                inputs.append(episode_based_inputs[stream_names_index])

        return inputs, outputs
