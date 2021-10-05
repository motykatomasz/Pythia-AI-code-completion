import argparse
import time
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.bpe_net import BPE_net
from datasets.Python150k_bpe import Python150k_bpe
from utils.Logging import AverageMeter, ProgressMeter
from utils.utils import print_size_of_model
import sentencepiece as spm
import sacrebleu

import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser(description='ML4SE project training script')
parser.add_argument('--dataset_path', metavar='path/to/python150k', default='data/BPE',
                    type=str, help='path to dataset')
parser.add_argument('--batch_size', metavar='5', default=64, type=int,
                    help='batch size')
parser.add_argument('--embed_dim', metavar='150', default=150, type=int,
                    help='dimension of the embedding')
parser.add_argument('--hidden_dim', metavar='100', default=500, type=int,
                    help='dimension of the LSTM hidden unit')
parser.add_argument('--num_layers', metavar='2', default=2, type=int,
                    help='number of LSTM layers')
parser.add_argument('--lookback_tokens', metavar='100', default=100, type=int,
                    help='number of lookback tokens')
parser.add_argument('--pin_memory', metavar='[True,False]', default=True,
                    type=bool, help='pin memory on GPU')
parser.add_argument('--num_workers', metavar='8', default=3, type=int,
                    help='number of dataloader workers')
parser.add_argument('--lr_init', metavar='1e-2', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_weight_decay', metavar='1e-4', default=0.97, type=float,
                    help='Weight decay per epoch (gamma).')
parser.add_argument('--l2_regularization', metavar='1e-4', default=1e-6, type=float,
                    help='weight decay for Adam optimizer (L2 regularization)')
parser.add_argument('--dropout', metavar='0.5', default=0.4, type=float,
                    help='dropout probability for LSTM')
parser.add_argument('--max_norm_grad', metavar='1', default=10, type=float,
                    help='maximum norm of gradients')
parser.add_argument('--epochs', metavar='5', default=10, type=int,
                    help='number of training epochs')
parser.add_argument('--predict', metavar='path/to/weights', default=None, type=str,
                    help='provide path to model weights to predict on validation set')
parser.add_argument('--conf_mat', metavar='false', default=False, type=bool,
                    help='True if you want to save the confusion matrix after each epoch '
                         '[not recommended for big vocab]')
parser.add_argument('--weighted_loss', metavar='path', default=None, type=str,
                    help='Path to weights for weighted loss. Set to None to not use weights.')
parser.add_argument('--model_name', metavar='model_bpe_final', default="best_model_bpe", type=str,
                    help='name or path of the pre-trained model')
parser.add_argument('--mode', metavar='model_final', default="bpe", type=str,
                    help='type of problem')
parser.add_argument('--max_len_label', metavar='model_final', default=20, type=int,
                    help='max allowed length of the label subtokens')


def test():
    global args
    args = parser.parse_args()
    saved_model = args.model_name

    testset = Python150k_bpe(args.dataset_path, mode='test', lookback_tokens=args.lookback_tokens, max_len_label=args.max_len_label)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(os.path.join(args.dataset_path, 'voc_bpe.model'))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=testset.padding_idx)

    # testset trainset vocab same right? check the passed arguments
    vocab_len = 10001
    padding_idx = 10000
    model = BPE_net(embedding_dim=args.embed_dim, vocab_size=vocab_len,
                  padding_idx=padding_idx, hidden_dim=args.hidden_dim,
                  batch_size=args.batch_size, num_layers=args.num_layers, dropout=args.dropout, max_label_len=args.max_len_label)


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


    batch_time = AverageMeter('Time', ':6.3f')
    top_1_acc_running = AverageMeter('Top-1-Accuracy', ':.3f')
    bleu_running = AverageMeter('Bleu', ':.3f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, top_1_acc_running, bleu_running],
        prefix="Test, epoch: [{}]".format("test"))

    model.eval()

    end = time.time()

    with torch.no_grad():
        for epoch_step, (input, input_len, label, label_len) in enumerate(testloader):
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            # eventhough we pass labels to the model, it is not being used with testing=True
            out, _ = model(input, input_len, label, label_len, testing=True)

            # Ignore prediction for padding token
            out = out[..., :vocab_len - 1].contiguous()
            # loss = criterion(out, label)

            # loss, acc = calc_loss_and_acc(out, label, label_len, criterion, sp_bpe)
            top_1_acc, bleu = evaluate_predictions(out, label, label_len, sp_bpe)

            # Statistics
            bleu_running.update(bleu, args.batch_size)
            top_1_acc_running.update(top_1_acc, args.batch_size)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()
            #

        return


def main():
    global args
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    trainset = Python150k_bpe(args.dataset_path, mode='train', lookback_tokens=args.lookback_tokens, max_len_label=args.max_len_label)
    valset = Python150k_bpe(args.dataset_path, mode='val', lookback_tokens=args.lookback_tokens, max_len_label=args.max_len_label)

    # Dataloaders
    dataloaders = dict()
    dataloaders['train'] = DataLoader(trainset,
                                    batch_size=args.batch_size, shuffle=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = DataLoader(valset,
                                    batch_size=args.batch_size, shuffle=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers)


    vocab_len = 10001
    padding_idx = 10000

    net = BPE_net(embedding_dim=args.embed_dim, vocab_size=vocab_len,
                               padding_idx=padding_idx, hidden_dim=args.hidden_dim,
                               batch_size=args.batch_size, num_layers=args.num_layers, dropout=args.dropout, max_label_len=args.max_len_label)


    print(torch.cuda.is_available())
    # Push the models to the GPU
    if torch.cuda.is_available():
        net = net.cuda()
        print('Models pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    print_size_of_model(net)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=trainset.padding_idx)
    optimizer = optim.Adam(net.parameters(), lr=args.lr_init, weight_decay=args.l2_regularization)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_weight_decay)

    # For storing the statistics
    metrics = {'train_loss': [],
               'train_acc_top_1': [],
               'val_acc_top_1': [],
               'val_loss': []}

    # Num of epochs for training
    num_epochs = args.epochs

    best_model = None
    best_acc = 0

    # Training loop
    for epoch in range(num_epochs):

        train_loss, train_acc = train_epoch(dataloaders['train'], net,
                                            criterion, optimizer, scheduler, epoch,
                                            vocab_len)

        metrics['train_loss'].append(train_loss)
        metrics['train_acc_top_1'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc = validate_epoch(dataloaders['val'], net,
                                            criterion, epoch,
                                            vocab_len)

        metrics['val_loss'].append(val_loss)
        metrics['val_acc_top_1'].append(val_acc)
        print('Epoch {} val loss: {:.4f}, acc: {:.4f}'.format(epoch, val_loss, val_acc))

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

        if val_acc > best_acc:
            best_model = net.state_dict()
            torch.save(best_model, f"best_model_bpe")

    torch.save(net.state_dict(), f"model_bpe_final")


def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, vocab_len):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    end = time.time()

    model.train()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        # for epoch_step, (input, input_len, label) in enumerate(dataloader):
        for epoch_step, (input, input_len, label, label_len) in enumerate(dataloader):

            optimizer.zero_grad()

            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out, _ = model(input, input_len, label, label_len)

            # Ignore prediction for padding token
            out = out[..., :vocab_len-1]

            # Optimizer
            loss, acc = calc_loss_and_acc(out, label, label_len, criterion)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_norm_grad)
            optimizer.step()

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            # Accuracy
            acc_running.update(acc, bs)

            # output training info
            progress.display(epoch_step)
            #
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step()

    return loss_running.avg, acc_running.avg


def validate_epoch(dataloader, model, criterion, epoch, vocab_len):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Validate, epoch: [{}]".format(epoch))

    end = time.time()

    model.eval()

    with torch.no_grad():
        # Iterate over data.
        # for epoch_step, (input, input_len, label) in enumerate(dataloader):
        for epoch_step, (input, input_len, label, label_len) in enumerate(dataloader):

            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            out, _ = model(input, input_len, label, label_len)

            # Ignore prediction for padding token
            out = out[..., :vocab_len-1]

            # Optimizer
            loss, acc = calc_loss_and_acc(out, label, label_len, criterion)

            # Statistics
            bs = input.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            # Accuracy
            acc_running.update(acc, bs)

            # output training info
            progress.display(epoch_step)
            #
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

    return loss_running.avg, acc_running.avg


def calc_loss_and_acc(out, label, label_len, criterion):
    losses = []
    subtokens_predicted = 0
    total_subtokens = 0
    for i, sample in enumerate(out):
        p = out[i].permute(1, 0)[..., :label_len[i]].permute(1, 0)
        l = label[i][:label_len[i]]

        losses.append(criterion(p, l))

        pred = p.argmax(dim=1)
        subtokens_predicted += torch.sum(pred == l)
        total_subtokens += l.shape[0]

    loss = sum(losses)
    acc = subtokens_predicted.item()/total_subtokens

    return loss, acc


def evaluate_predictions(out, label, label_len, sp):
    total_bleu = 0
    EOF_index = 2
    predicted = 0
    for i, sample in enumerate(out):
        # select the token with maximum probability value
        _, next_word = torch.max(out[i], dim=1)
        output_sequence = next_word.cpu().numpy().tolist()
        output_sequence = [int(i) for i in output_sequence if int(i) != EOF_index]
        # Use sentence piece function to get the predicted label
        predicted_label = sp.decode_ids(output_sequence)

        # Now get teh true label
        l = label[i][:label_len[i]]
        l = l.cpu().numpy().tolist()
        l = [int(i) for i in l]
        true_label = sp.decode_ids(l)

        # maybe average the bleu score or something...
        bleu_score = sacrebleu.corpus_bleu(true_label, predicted_label).score
        # print(bleu_score)
        total_bleu += bleu_score

        if predicted_label == true_label:
            predicted += 1



    return predicted/out.size(0), total_bleu/out.size(0)


if __name__ == "__main__":
    # main()
    test()
    # print(test())
