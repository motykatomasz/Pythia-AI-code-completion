import argparse
import time
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.code_completion_network import NeuralCodeCempletion
from models.code_completion_network_attention import NeuralCodeCompletionAttention
from datasets.Python150k import Python150k
from utils.Logging import AverageMeter, ProgressMeter
from utils.utils import save_confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

# Parse input arguments
parser = argparse.ArgumentParser(description='ML4SE project training script')
parser.add_argument('--dataset_path', metavar='path/to/python150k', default='data',
                    type=str, help='path to dataset')
parser.add_argument('--batch_size', metavar='5', default=64, type=int,
                    help='batch size')
parser.add_argument('--embed_dim', metavar='150', default=150, type=int,
                    help='dimension of the embedding')
parser.add_argument('--hidden_dim', metavar='100', default=500, type=int,
                    help='dimension of the LSTM hidden unit')
parser.add_argument('--num_layers', metavar='2', default=2, type=int,
                    help='number of LSTM layers')
parser.add_argument('--lookback_tokens', metavar='100', default=1000, type=int,
                    help='number of lookback tokens')
parser.add_argument('--pin_memory', metavar='[True,False]', default=True,
                    type=bool, help='pin memory on GPU')
parser.add_argument('--num_workers', metavar='8', default=24, type=int,
                    help='number of dataloader workers')
parser.add_argument('--lr_init', metavar='1e-2', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_weight_decay', metavar='1e-4', default=0.97, type=float,
                    help='Weight decay per epoch (gamma).')
parser.add_argument('--l2_regularization', metavar='1e-4', default=0, type=float,
                    help='weight decay for Adam optimizer (L2 regularization)')
parser.add_argument('--dropout', metavar='0.5', default=0, type=float,
                    help='dropout probability for LSTM')
parser.add_argument('--max_norm_grad', metavar='1', default=5, type=float,
                    help='maximum norm of gradients')
parser.add_argument('--epochs', metavar='5', default=30, type=int,
                    help='number of training epochs')
parser.add_argument('--predict', metavar='path/to/weights', default=None, type=str,
                    help='provide path to model weights to predict on validation set')
parser.add_argument('--conf_mat', metavar='false', default=False, type=bool,
                    help='True if you want to save the confusion matrix after each epoch '
                         '[not recommended for big vocab]')
parser.add_argument('--weighted_loss', metavar='path', default=None, type=str,
                    help='Path to weights for weighted loss. Set to None to not use weights.')
parser.add_argument('--model_name', metavar='model_final', default="model_epoch_6_acc_0.631", type=str,
                    help='name or path of the pre-trained model')
parser.add_argument('--max_len_label', metavar='model_final', default=20, type=int,
                    help='type of problem')

def test():
    global args
    args = parser.parse_args()
    saved_model = args.model_name

    testset = Python150k(args.dataset_path, mode='test', lookback_tokens=args.lookback_tokens)

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

    writer = SummaryWriter('runs')

    word2idx, idx2word, type2dix, idx2type = testset.get_mappings()

    if args.weighted_loss is not None:
        weights = torch.FloatTensor(np.load(args.weighted_loss)).cuda()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=testset.padding_idx, weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=testset.padding_idx)

    # testset trainset vocab same right? check the passed arguments
    model = NeuralCodeCempletion(embedding_dim=args.embed_dim, vocab_size=testset.get_vocab_len(),
                               padding_idx=testset.padding_idx, hidden_dim=args.hidden_dim,
                               batch_size=args.batch_size, num_layers=args.num_layers, dropout=args.dropout)

    # Code completion network WITH ATTENTION
    # model = NeuralCodeCompletionAttention(embedding_dim=args.embed_dim, vocab_size=trainset.get_vocab_len(),
    #                            padding_idx=trainset.padding_idx, hidden_dim=args.hidden_dim,
    #                            batch_size=args.batch_size, num_layers=args.num_layers, dropout=0)

    if os.path.exists(os.getcwd() + "/" + saved_model):
        print(f"Loading pre-trained weights from file: {saved_model}")
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), saved_model)))
    else:
        print(f"Model by the name: {saved_model} NOT FOUND")

    print(torch.cuda.is_available())
    # Push the models to the GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print('Models pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    vocab_len = testset.get_vocab_len()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Top-1 Accuracy', ':.3f')
    top5_acc_running = AverageMeter('Top-5 Accuracy', ':.3f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, data_time, loss_running, acc_running, top5_acc_running],
        prefix="Test, batch: [{}]".format("test"))

    model.eval()

    end = time.time()

    with torch.no_grad():
        for epoch_step, (input, input_len, label) in enumerate(testloader):
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out = model(input, input_len)

            # Ignore prediction for padding token
            out = out[..., :vocab_len - 1].contiguous()
            loss = criterion(out, label)

            top1_corrects, top5_corrects = calc_acc(input, label, input_len, out, type2dix, idx2type)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            # Top-1 accuracy
            acc = top1_corrects / bs
            acc_running.update(acc, bs)

            # Top-5 accuracy
            top5_acc = top5_corrects / bs
            top5_acc_running.update(top5_acc, bs)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

    return loss_running.avg, acc_running.avg, top5_acc_running.avg


def main():
    global args
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    trainset = Python150k(args.dataset_path, mode='train', lookback_tokens=args.lookback_tokens)
    valset = Python150k(args.dataset_path, mode='val', lookback_tokens=args.lookback_tokens)


    # # Create train, val and test datasets
    trainset = Python150k(args.dataset_path, mode='train', lookback_tokens=args.lookback_tokens)
    valset = Python150k(args.dataset_path, mode='val', lookback_tokens=args.lookback_tokens)

    # Dataloaders
    dataloaders = dict()
    dataloaders['train'] = DataLoader(trainset,
                                    batch_size=args.batch_size, shuffle=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = DataLoader(valset,
                                    batch_size=args.batch_size, shuffle=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers)


    writer = SummaryWriter('runs')

    # Code completion network
    net = NeuralCodeCempletion(embedding_dim=args.embed_dim, vocab_size=trainset.get_vocab_len(),
                               padding_idx=trainset.padding_idx, hidden_dim=args.hidden_dim,
                               batch_size=args.batch_size, num_layers=args.num_layers, dropout=args.dropout)

    # Code completion network WITH ATTENTION
    # net = NeuralCodeCompletionAttention(embedding_dim=args.embed_dim, vocab_size=trainset.get_vocab_len(),
    #                            padding_idx=trainset.padding_idx, hidden_dim=args.hidden_dim,
    #                            batch_size=args.batch_size, num_layers=args.num_layers, dropout=0)


    print(torch.cuda.is_available())
    # Push the models to the GPU
    if torch.cuda.is_available():
        net = net.cuda()
        print('Models pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    if args.weighted_loss is not None:
        weights = torch.FloatTensor(np.load(args.weighted_loss)).cuda()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=trainset.padding_idx, weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=trainset.padding_idx)

    optimizer = optim.Adam(net.parameters(), lr=args.lr_init, weight_decay=args.l2_regularization)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_weight_decay)

    # For storing the statistics
    metrics = {'train_loss': [],
               'train_acc_top_1': [],
               'train_acc_top_5': [],
               'val_acc_top_1': [],
               'val_loss': [],
               'val_acc_top_5': []}

    # Num of epochs for training
    num_epochs = args.epochs

    word2idx, idx2word, type2dix, idx2type = trainset.get_mappings()

    best_model = None
    best_acc = 0

    # Training loop
    for epoch in range(num_epochs):

        train_loss, train_acc, train_acc_5 = train_epoch(dataloaders['train'], net,
                                            criterion, optimizer, scheduler, epoch, type2dix, idx2type,
                                            trainset.get_vocab_len(), args.conf_mat)

        metrics['train_loss'].append(train_loss)
        metrics['train_acc_top_1'].append(train_acc)
        metrics['train_acc_top_5'].append(train_acc_5)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc, val_acc_5 = validate_epoch(dataloaders['val'], net,
                                            criterion, epoch, type2dix, idx2type,
                                            trainset.get_vocab_len(), args.conf_mat)
        metrics['val_acc_top_1'].append(val_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc_top_5'].append(val_acc_5)

        print('Top-1 accuracy      : {:5.3f}'.format(val_acc))
        print('Top-5 accuracy      : {:5.3f}'.format(val_acc_5))
        print('---------------------')

        with open(f"stats_{epoch}.txt", "w+") as f:
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    print(f"{key} - {value.item()}")
                    f.write(f"{key} - {value.item()}\n")
                else:
                    print(f"{key} - {value}")
                    f.write(f"{key} - {value}\n")
            f.flush()
            f.close()

        if val_acc_5 > best_acc:
            best_model = net.state_dict()

        top_5_accuracy = '{:5.3f}'.format(val_acc_5)
        torch.save(net.state_dict(), f"model_epoch_{epoch}_acc_{top_5_accuracy}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

    torch.save(net.state_dict(), f"model_final")
    torch.save(best_model, f"model_final")
    writer.flush()
    writer.close()


def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, type2idx, idx2type, vocab_len, cm):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    top1_acc_running = AverageMeter('Top-1 Accuracy', ':.3f')
    top5_acc_running = AverageMeter('Top-5 Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, top1_acc_running, top5_acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    end = time.time()

    model.train()

    if cm:
        epoch_cm = np.zeros(shape=(vocab_len, vocab_len))

    with torch.set_grad_enabled(True):
        # Iterate over data.
        # for epoch_step, (input, input_len, label) in enumerate(dataloader):
        for epoch_step, (input, input_len, label, label_len) in enumerate(dataloader):

            optimizer.zero_grad()

            # for comparing on token level using sentencepiece function?
            # for label_enc in label:
            #     print(label_enc)

            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out, _ = model(input, input_len, label, label_len)
            # out = model(input, input_len)
            if cm:
                epoch_cm = epoch_cm + confusion_matrix(label.cpu(), out.argmax(dim=1).cpu(),
                                                   labels=[i for i in range(vocab_len)])


            # Ignore prediction for padding token
            out = out[..., :vocab_len-1].contiguous()

            # Optimizer
            loss = criterion(out, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_norm_grad)
            optimizer.step()

            top1_corrects, top5_corrects = calc_acc(input, label, input_len, out, type2idx, idx2type)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            # Top-1 accuracy
            top1_acc = top1_corrects / bs
            top1_acc_running.update(top1_acc, bs)

            # Top-5 accuracy
            top5_acc = top5_corrects / bs
            top5_acc_running.update(top5_acc, bs)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step()
        if cm:
            save_confusion_matrix(epoch_cm, epoch, 'train')

    return loss_running.avg, top1_acc_running.avg, top5_acc_running.avg


def validate_epoch(dataloader, model, criterion, epoch, idx2word, idx2type, vocab_len, cm):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Top-1 Accuracy', ':.3f')
    top5_acc_running = AverageMeter('Top-5 Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running, top5_acc_running],
        prefix="Validate, epoch: [{}]".format(epoch))

    # Set model in evaluation mode
    model.eval()
    if cm:
        epoch_cm = np.zeros(shape=(vocab_len, vocab_len))

    with torch.no_grad():
        end = time.time()
        for epoch_step, (input, input_len, label) in enumerate(dataloader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out = model(input, input_len)

            if cm:
                epoch_cm = epoch_cm + confusion_matrix(label.cpu(), out.argmax(dim=1).cpu(),
                                                   labels=[i for i in range(vocab_len)])

            # Ignore prediction for padding token
            out = out[..., :vocab_len-1].contiguous()
            loss = criterion(out, label)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            top1_corrects, top5_corrects = calc_acc(input, label, input_len, out, idx2word, idx2type)

            # Top-1 accuracy
            acc = top1_corrects / bs
            acc_running.update(acc, bs)

            # Top-5 accuracy
            top5_acc = top5_corrects / bs
            top5_acc_running.update(top5_acc, bs)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

    if cm:
        save_confusion_matrix(epoch_cm, epoch, 'val')

    return loss_running.avg, acc_running.avg, top5_acc_running.avg


def calc_acc(input, label, input_len, out, type2idx, idx2type):
    """Calculates accuracy based on the matching types."""
    top_1_correct = 0
    top_5_correct = 0
    for i, sample in enumerate(input):
        type = idx2type[int(label[i])]
        last_token_type = idx2type[int(sample[int(input_len[i])-2])]

        # Type of last token doesn't match type of label
        if type is None or last_token_type != type:
            index = out[i].argmax()
            if index == label.data[i]:
                top_1_correct += 1

            values, indices = out[i].topk(k=5)
            if label.data[i] in indices:
                top_5_correct += 1

        # Type of last token matches type of label
        else:
            out_narrowed = out[i][type2idx[type]]
            index = out_narrowed.argmax()
            original_index = type2idx[type][index]
            if original_index == label.data[i]:
                top_1_correct += 1

            k = 5
            l = len(out_narrowed)
            if l < 5:
                k = l
            values, indices = out_narrowed.topk(k=k)
            original_indices = [type2idx[type][i] for i in indices]
            if label.data[i] in original_indices:
                top_5_correct += 1

    return top_1_correct, top_5_correct


def calc_loss_and_acc(input, label, input_len, out, type2idx, idx2type, loss):
    """
    Calculates loss and accuracy based on the matching types.
        CAN'T BE USED WITH WEIGHTED LOSS.
    """
    top_1_correct = 0
    top_5_correct = 0
    acc_loss = 0
    for i, sample in enumerate(input):
        type = idx2type[int(label[i])]
        last_token_type = idx2type[int(sample[int(input_len[i])-2])]

        # Type of last token doesn't match type of label
        if type is None or last_token_type != type:
            index = out[i].argmax()
            if index == label.data[i]:
                top_1_correct += 1

            values, indices = out[i].topk(k=5)
            if label.data[i] in indices:
                top_5_correct += 1

            acc_loss += loss(out[i].view(1, -1), label[i].view(1))

        # Type of last token matches type of label
        else:
            out_narrowed = out[i][type2idx[type]]
            index = out_narrowed.argmax()
            original_index = type2idx[type][index]
            if original_index == label.data[i]:
                top_1_correct += 1
                acc_loss += loss(out_narrowed.view(1, -1), torch.LongTensor([index]).cuda())
            else:
                acc_loss += loss(out[i].view(1, -1), label[i].view(1))

            k = 5
            l = len(out_narrowed)
            if l < 5:
                k = l
            values, indices = out_narrowed.topk(k=k)
            original_indices = [type2idx[type][i] for i in indices]
            if label.data[i] in original_indices:
                top_5_correct += 1

    return acc_loss/len(label), top_1_correct, top_5_correct


if __name__ == "__main__":
    main()
    # print(test())
