# coding=utf-8
"""Code by Noa Garcia and Yuta Nakashima"""
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import LayerNorm, Dropout
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils
from fusion_data_sample import FusionDataSample
from multi_head_attention import MultiHeadAttention
from utils import create_folder_with_timestamp, str2bool

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True, type=bool)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nepochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--no_cuda', action='store_true')

    """Code by InterDigital"""
    parser.add_argument("--num_max_slices_plot", default=10, type=int)
    parser.add_argument("--num_max_slices_episode_dialog_summary", default=10, type=int)
    parser.add_argument('--fuse_stream_list', nargs='+', required=True, type=str)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument("--load_pretrained_model_exists", default=False, type=bool)
    parser.add_argument('--ss_max_temperature', default=2, type=int)
    parser.add_argument('--lr_patience', default=5, type=int)
    parser.add_argument('--fusion_method', required=True, type=str)
    parser.add_argument("--fusion_train_folder_path", default='Training', type=str)
    parser.add_argument("--stream_train_folder_path", default='Training/', type=str)
    parser.add_argument("--pretrain_modeldir", type=str)
    parser.add_argument("--part_selection_with_soft_temporal_attention", default=True, type=str2bool)
    parser.add_argument("--save_multi_stream_attention_scores", default=False, type=str2bool)

    args, unknown = parser.parse_known_args()
    return args

class TwoInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MultiStreamAttention(nn.Module):
    def __init__(self, input_shape):
        super(MultiStreamAttention, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_shape, input_shape // 2), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(input_shape // 2, 1), nn.Dropout(0.5),
                                    nn.Softmax(dim=1))

    def forward(self, input):
        score = self.layers(input)

        input_ = input * score

        if save_multi_stream_scores:
            attention_scores.append(score)

        return input_


class ResidualSelfAttention(nn.Module):
    def __init__(self, embed, num_head):
        super(ResidualSelfAttention, self).__init__()

        self.layer = MultiHeadAttention(embed, num_head)
        self.norm = LayerNorm(embed)
        self.dropout = Dropout(0.5)

    def forward(self, input):
        attended = self.layer(input, input, input)[0]
        output = self.norm(self.dropout(attended) + input)
        return output


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return torch.flatten(input,1)


class FusionProduct(nn.Module):
    def __init__(self, args):
        self.args = args
        super(FusionProduct, self).__init__()

        self.stream_transformer_blocks = nn.ModuleList([nn.Sequential(Flatten(),
                                                                      nn.Linear(768, 1)) for _ in range(4)])

    def forward(self, inputs):

        num_choices = inputs[0].shape[1]

        reshaped_scores_list = []
        score_list = []

        for choice, module_per_answer in zip(range(num_choices), self.stream_transformer_blocks):

            result = torch.ones_like(inputs[0][:, 0, :])

            for br in inputs:
                result = br[:, choice, :] * result

            answer = module_per_answer(result)
            score_list.append(answer)

        # Final score
        all_feat = torch.squeeze(torch.cat(score_list, 1), 1)

        reshaped_final_scores = all_feat.view(-1, num_choices)
        return reshaped_scores_list, reshaped_final_scores


class FusionMethods(nn.Module):
    def __init__(self, args):
        self.args = args
        super(FusionMethods, self).__init__()

        if args.fusion_method == 'multi-stream-attention':
            self.stream_transformer_blocks = nn.ModuleList([nn.Sequential(MultiStreamAttention(768), Flatten(),
                                                                          nn.Linear(768 * len(args.fuse_stream_list),
                                                                                    1)) for _ in range(4)])


        elif args.fusion_method == 'self-attention':
            self.stream_transformer_blocks = nn.ModuleList(
                [nn.Sequential(ResidualSelfAttention(768, args.num_head),ResidualSelfAttention(768, args.num_head),
                               Flatten(),
                               nn.Linear(768 * len(args.fuse_stream_list), 1)) for _ in range(4)])

        elif args.fusion_method == 'multi-stream-self-attention':
            self.stream_transformer_blocks = nn.ModuleList(
                [nn.Sequential(MultiStreamAttention(768), ResidualSelfAttention(768, args.num_head), Flatten(),
                               nn.Linear(768 * len(args.fuse_stream_list), 1)) for _ in range(4)])

        elif args.fusion_method == 'product':
            pass

        else:
            raise NotImplementedError

    def forward(self, inputs):

        num_choices = inputs[0].shape[1]

        reshaped_scores_list = []
        score_list = []

        for choice, module_per_answer in zip(range(num_choices), self.stream_transformer_blocks):

            if self.args.fusion_method == 'product':
                result = torch.ones_like(inputs[0][:, 0, :])
                for br in inputs:
                    result = br[:, choice, :] * result
                answer = module_per_answer(result)
                score_list.append(answer)
            else:
                answer_list = []
                for br in inputs:
                    answer_list.append(br[:, choice, :])
                stack = torch.stack(answer_list, dim=1)
                answer = module_per_answer(stack)
                score_list.append(answer)

        # Final score
        all_feat = torch.squeeze(torch.cat(score_list, 1), 1)

        reshaped_final_scores = all_feat.view(-1, num_choices)
        return reshaped_scores_list, reshaped_final_scores

"""Code by Noa Garcia and Yuta Nakashima"""
def trainEpoch(args, train_loader, model, criterion, optimizer, epoch):
    losses = utils.AverageMeter()
    model.train()

    targets = []
    outs = []

    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        output, final_scores = model(input_var)

        # Compute loss
        final_loss = criterion(final_scores, target_var[0])

        train_loss = final_loss

        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # for plot
        outs.append(torch.max(final_scores, 1)[1].data.cpu().numpy())
        targets.append(target[0].cpu().numpy())

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

    outs = np.concatenate(outs).flatten()
    targets = np.concatenate(targets).flatten()

    acc = np.sum(outs == targets) / len(outs)
    return epoch, losses.avg, acc, None


def valEpoch(args, val_loader, model, criterion, epoch):
    losses = utils.AverageMeter()
    # final_losses = utils.AverageMeter()
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output, final_scores = model(input_var)

        # Compute loss
        predicted = torch.max(final_scores, 1)[1]

        stream_predictions = [torch.max(p, 1)[1] for p in output]

        final_loss = criterion(final_scores, target_var[0])
        train_loss = 0

        train_loss = final_loss

        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Save predictions to compute accuracy
        if batch_idx == 0:
            out = predicted.data.cpu().numpy()
            out_stream_list = []
            for p in stream_predictions:
                out_stream_list.append(p.data.cpu().numpy())
            label = target[0].cpu().numpy()
        else:
            out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)
            label = np.concatenate((label, target[0].cpu().numpy()), axis=0)
            for idx in range(len(stream_predictions)):
                out_stream_list[idx] = np.concatenate(
                    (out_stream_list[idx], stream_predictions[idx].data.cpu().numpy()), axis=0)

    # Accuracy
    acc = np.sum(out == label) / len(out)
    logger.info('Validation set: Average loss: {:.4f}\t'
                'Accuracy {acc}'.format(losses.avg, acc=acc))

    return epoch, losses.avg, acc, None


def train(args, modeldir):
    # Set GPU
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Model, optimizer and loss

    if args.fusion_method == "product":
        model = FusionProduct(args)
    else:
        model = FusionMethods(args)
    if args.device == "cuda":
        model.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    class_loss = nn.CrossEntropyLoss().cuda()

    # Data
    trainDataObject = FusionDataSample(args, split='train')
    valDataObject = FusionDataSample(args, split='val')
    train_loader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(valDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                             num_workers=args.workers)

    # Now, let's start the training process!
    logger.info('Training loader with %d samples' % train_loader.__len__())
    logger.info('Validation loader with %d samples' % val_loader.__len__())
    logger.info('Training...')
    pattrack = 0
    best_val = 0


    for epoch in range(0, args.nepochs):

        # Epoch
        trainEpoch(args, train_loader, model, class_loss, optimizer, epoch)
        epoch_plot_val, loss_plot_val, acc_plot_val, stream_losses_plot_val = valEpoch(args, val_loader, model,
                                                                                       class_loss, epoch)
        current_val = acc_plot_val

        scheduler.step(loss_plot_val)

        # Check patience
        is_best = current_val > best_val
        best_val = max(current_val, best_val)
        if not is_best:
            pattrack += 1
        else:
            pattrack = 0
        if pattrack >= args.patience:
            break

        logger.info('** Validation information: %f (this accuracy) - %f (best accuracy) - %d (patience valtrack)' % (
            current_val, best_val, pattrack))

        # Save
        state = {'state_dict': model.state_dict(),
                 'best_val': best_val,
                 'optimizer': optimizer.state_dict(),
                 'pattrack': pattrack,
                 'curr_val': current_val}
        filename = os.path.join(modeldir, 'model_latest.pth.tar')
        torch.save(state, filename)
        if is_best:
            filename = os.path.join(modeldir, 'model_best.pth.tar')
            torch.save(state, filename)


def evaluate(args, modeldir):
    n_gpu = torch.cuda.device_count()

    """Code by InterDigital"""
    if args.fusion_method == "product":
        model = FusionProduct(args)
    else:
        model = FusionMethods(args)

    """Code by Noa Garcia and Yuta Nakashima"""
    if args.device == "cuda":
        model.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("=> loading checkpoint from '{}'".format(modeldir))
    checkpoint = torch.load(os.path.join(modeldir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Data
    evalDataObject = FusionDataSample(args, split='test')
    test_loader = torch.utils.data.DataLoader(evalDataObject, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=(not args.no_cuda), num_workers=args.workers)
    logger.info('Evaluation loader with %d samples' % test_loader.__len__())

    # Switch to evaluation mode & compute test samples embeddings
    batch_time = utils.AverageMeter()
    end = time.time()
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output, final_scores = model(input_var)
        # Compute final loss
        predicted = torch.max(final_scores, 1)[1]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Store outputs
        if i == 0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
            index = target[1].cpu().numpy()

            score_list = []
            for o in output:
                score_list.append(o.data.cpu().numpy())
            scores_final = final_scores.data.cpu().numpy()
        else:
            out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)
            label = np.concatenate((label, target[0].cpu().numpy()), axis=0)
            index = np.concatenate((index, target[1].cpu().numpy()), axis=0)

            for idx in range(len(score_list)):
                score_list[idx] = np.concatenate((score_list[idx], output[idx].data.cpu().numpy()), axis=0)

            scores_final = np.concatenate((scores_final, final_scores.cpu().numpy()), axis=0)

    df = pd.read_csv(os.path.join(args.data_dir, 'knowit_data_test.csv'), delimiter='\t')

    logger.info("Eval on test data from fusion final output")

    """Code by InterDigital"""
    with open(os.path.join(modeldir, 'test_results_fusion.npy'), 'wb') as f:
        np.save(f, out)
    with open(os.path.join(modeldir, 'test_labels_fusion.npy'), 'wb') as f:
        np.save(f, label)
    """Code by Noa Garcia and Yuta Nakashima"""
    utils.accuracy(df, out, label, index)


if __name__ == "__main__":

    args = get_params()

    """Code by InterDigital"""
    assert len(args.fuse_stream_list) >= 2  # Make sure at least two streams given

    if args.load_pretrained_model_exists:
        if args.pretrain_modeldir is not None:
            modeldir = args.pretrain_modeldir
        else:
            raise FileNotFoundError
    else:
        # Create training and data directories
        modeldir = create_folder_with_timestamp(os.path.join(args.fusion_train_folder_path, "-".join(args.fuse_stream_list)+'_'+args.fusion_method),
                                                args.load_pretrained_model_exists)

    global attention_scores
    if args.save_multi_stream_attention_scores:
        attention_scores = []

    global save_multi_stream_scores
    save_multi_stream_scores = args.save_multi_stream_attention_scores


    args.modeldir = modeldir

    logger.info("Arguments: %s" % json.JSONEncoder().encode(vars(args)))

    with open(os.path.join(modeldir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    """Code by Noa Garcia and Yuta Nakashima"""
    # Train if model does not exist
    if not os.path.isfile(os.path.join(modeldir, 'model_best.pth.tar')):
        train(args, modeldir)

    # Evaluation
    evaluate(args, modeldir)

    """Code by InterDigital"""
    if args.save_multi_stream_attention_scores:
        with open(os.path.join(modeldir, "test_soft_attention_score.npy"), "wb") as f:
            np.save(f, np.array([a.cpu().numpy() for a in attention_scores]))
