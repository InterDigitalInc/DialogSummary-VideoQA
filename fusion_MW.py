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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils
from fusion_data_sample import FusionDataSample
from utils import create_folder_with_timestamp, str2bool

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nepochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument("--num_max_slices_plot", default=10, type=int)
    parser.add_argument("--num_max_slices_episode_dialog_summary", default=10, type=int)
    parser.add_argument('--weight_loss_final', default=0.7, type=float)

    """Code by InterDigital"""
    parser.add_argument('--fuse_stream_list', nargs='+', required=True, type=str)
    parser.add_argument('--fuse_loss_weight_list', nargs='+', default=None, type=float)
    parser.add_argument("--load_pretrained_model_exists", default=False)
    parser.add_argument("--eval_split", default="test", type=str)
    parser.add_argument('--lr_patience', default=5, type=int)
    parser.add_argument("--stream_train_folder_path", default='Training/main_stream_trainings', type=str)
    parser.add_argument("--fusion_train_folder_path", default='Training/fusion', type=str)
    parser.add_argument("--part_selection_with_soft_temporal_attention", default=True, type=str2bool)
    parser.add_argument('--ss_max_temperature', default=2, type=int)
    args, unknown = parser.parse_known_args()
    return args


class FusionMW(nn.Module):
    def __init__(self, args):
        self.args = args
        super(FusionMW, self).__init__()

        number_of_streams = len(args.fuse_loss_weight_list)
        self.module_list = nn.ModuleList([nn.Sequential(nn.Linear(768, 1)) for _ in range(number_of_streams)])

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(number_of_streams, 1))

    def forward(self, inputs):
        assert len(self.module_list) == len(inputs)
        num_choices = inputs[0].shape[1]
        reshaped_scores_list = []
        score_list = []
        for module_per_stream, input_per_stream in zip(self.module_list, inputs):
            flat_in = input_per_stream.view(-1, input_per_stream.size(-1))
            score = module_per_stream(self.dropout(flat_in))
            score_list.append(score)
            """Code by Noa Garcia and Yuta Nakashima"""
            reshaped_scores_list.append(score.view(-1, num_choices))

        # Final score
        all_feat = torch.squeeze(torch.cat(score_list, 1), 1)
        final_scores = self.classifier(all_feat)
        reshaped_final_scores = final_scores.view(-1, num_choices)
        return reshaped_scores_list, reshaped_final_scores


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
        train_loss = 0

        """Code by InterDigital"""
        for idx in range(len(output)):
            stream_loss = criterion(output[idx], target_var[0])
            # Track loss
            train_loss += args.fuse_loss_weight_list[idx] * stream_loss
        train_loss += final_loss * args.weight_loss_final

        """Code by Noa Garcia and Yuta Nakashima"""
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

        final_loss = criterion(final_scores, target_var[0]) * args.weight_loss_final
        train_loss = 0

        """Code by InterDigital"""
        for idx in range(len(output)):
            weighted_stream_loss = args.fuse_loss_weight_list[idx] * criterion(output[idx], target_var[0])
            train_loss += weighted_stream_loss
        train_loss += final_loss

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


    """Code by Noa Garcia and Yuta Nakashima"""
    # Accuracy
    acc = np.sum(out == label) / len(out)
    logger.info('Validation set: Average loss: {:.4f}\t'
                'Accuracy {acc}'.format(losses.avg, acc=acc))

    logger.info('Acc Streams: %s' % [a + ": " + str(b) for a, b in
                                     zip(args.fuse_stream_list,
                                         [(np.sum(o == label) / len(o)) for o in out_stream_list])])

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

    model = FusionMW(args)
    if args.device == "cuda":
        model.cuda()
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

    logger.info('Training loader with %d samples' % train_loader.__len__())
    logger.info('Validation loader with %d samples' % val_loader.__len__())
    logger.info('Training...')
    pattrack = 0
    best_val = 0

    for epoch in range(0, args.nepochs):

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
    model = FusionMW(args)
    if args.device == "cuda":
        model.cuda()
    logger.info("=> loading checkpoint from '{}'".format(modeldir))
    checkpoint = torch.load(os.path.join(modeldir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Data
    evalDataObject = FusionDataSample(args, split=args.eval_split)
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

    df = pd.read_csv(os.path.join(args.data_dir, 'knowit_data_%s.csv' % args.eval_split), delimiter='\t')

    """Code by InterDigital"""
    logger.info("Eval on %s data from fusion final output" % args.eval_split)
    if args.eval_split == 'test':
        utils.accuracy(df, out, label, index)
        logger.info("Eval on %s data from streams" % args.eval_split)
        for o, str_stream in zip(score_list, args.fuse_stream_list):
            logger.info("Stream: %s" % str_stream)
            utils.accuracy(df, np.argmax(o, 1), label, index)
    else:
        """Code by Noa Garcia and Yuta Nakashima"""
        utils.accuracy_val(out, label)


if __name__ == "__main__":

    args = get_params()
    """Code by InterDigital"""
    assert (args.fuse_loss_weight_list is not None) or (
            args.weight_loss_final is not None)  # At least one loss weight should be given

    if args.weight_loss_final is None:
        args.weight_loss_final = 1 - sum(args.fuse_loss_weight_list)
    elif args.fuse_loss_weight_list is None:
        remaining_loss_weight = 1 - args.weight_loss_final
        args.fuse_loss_weight_list = [remaining_loss_weight / len(args.fuse_stream_list)] * len(args.fuse_stream_list)

    assert len(args.fuse_stream_list) >= 2  # Make sure at least two streams given
    assert len(args.fuse_stream_list) == len(
        args.fuse_loss_weight_list)  # Make sure to give loss weight in the same amount of  streams
    assert sum(args.fuse_loss_weight_list) + args.weight_loss_final < 1.1  # Normalize the loss two approx. 1
    assert sum(args.fuse_loss_weight_list) + args.weight_loss_final > 0.9



    model_name_path = os.path.join(args.fusion_train_folder_path, "-".join(args.fuse_stream_list)+"_"+str(args.weight_loss_final))

    modeldir = create_folder_with_timestamp(model_name_path,
                                            args.load_pretrained_model_exists)

    logger.info("Arguments: %s" % json.JSONEncoder().encode(vars(args)))

    with open(os.path.join(modeldir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    """Code by Noa Garcia and Yuta Nakashima"""
    # Train if model does not exist
    if not os.path.isfile(os.path.join(modeldir, 'model_best.pth.tar')):
        train(args, modeldir)

    # Evaluation
    evaluate(args, modeldir)