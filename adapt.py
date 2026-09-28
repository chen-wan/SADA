"""Adversarial adaptation to train target encoder."""
import copy
import itertools
import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch import nn
import params
from evaluate import evaluate


def train_trg(src_encoder, tgt_encoder, critic, src_classifier,
              src_data_loader, tgt_data_loader, val_loader, tgt_test, patience, epochs):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    criticA=critic
    criticB=critic

    # setup criterion and optimizer
    criterion_disc = nn.CrossEntropyLoss()
    criterion_tgt = nn.CrossEntropyLoss(reduction='none')

    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate * 0.1,
                               betas=(params.beta1, params.beta2))
    optimizer_criticA = optim.Adam(criticA.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))

    optimizer_criticB = optim.Adam(criticB.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))

    ####################
    # 2. train network #
    ####################
    loss_val, acc_val, prob_val, label_val, _ = evaluate(tgt_encoder, src_classifier, val_loader)
    acc_best = acc_val
    trg_encoder_best = copy.deepcopy(tgt_encoder.state_dict())

    num_patience = 0
    Loss_val = []
    Acc_val = []


    # for param in tgt_encoder.parameters():
    #     param.requires_grad = False
    #
    # for layer in tgt_encoder.lastLayer.parameters():
    #     layer.requires_grad = True

    for epoch in range(epochs):

        # set train state for Dropout and BN layers
        tgt_encoder.train()
        criticA.train()  #
        criticB.train()

        # zip source and target data pair，
        for step, ((images_src, labels_src), (images_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable，
            images_src, labels_src = images_src.cuda(), labels_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for optimizer，
            optimizer_criticA.zero_grad()
            optimizer_criticB.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            output = src_classifier(feat_tgt)
            soft_output = nn.Softmax(dim=1)(output)
            tgt_pre = soft_output.max(dim=1)[1]

            loss_discA = loss_disc(feat_src, feat_tgt, labels_src, tgt_pre, criticA, criterion_disc, nclass=0)
            loss_discA.backward()  #

            # optimize critic，
            optimizer_criticA.step()  #

            loss_discB = loss_disc(feat_src, feat_tgt, labels_src, tgt_pre, criticA, criterion_disc, nclass=1)
            loss_discB.backward()  #

            # optimize critic，
            optimizer_criticB.step()

            ############################
            # 2.2 train target encoder #
            ############################
            # zero gradients for optimizer，
            optimizer_criticA.zero_grad()
            optimizer_criticB.zero_grad()
            optimizer_tgt.zero_grad()

            output = src_classifier(feat_tgt)
            soft_output = nn.Softmax(dim=1)(output)
            tgt_pre = soft_output.max(dim=1)[1]

            if torch.sum(tgt_pre==0).item()==0:
                loss_advA = 0
            else:
                feat_tgtA = feat_tgt[tgt_pre==0, :]
                soft_outputA = soft_output[tgt_pre==0, :]
                loss_advA = loss_adv(feat_tgtA, soft_outputA, criticA, criterion_tgt)

            if torch.sum(tgt_pre==1).item()==0:
                loss_advB = 0
            else:
                feat_tgtB = feat_tgt[tgt_pre==1, :]
                soft_outputB = soft_output[tgt_pre==1, :]
                loss_advB = loss_adv(feat_tgtB, soft_outputB, criticB, criterion_tgt)

            # compute loss for target encoder，
            loss_tgt = (loss_advA + loss_advB)*0.5

            loss_tgt.backward()

            # optimize target encoder，
            optimizer_tgt.step()

        loss_val, acc_val, prob_val, label_val, _ = evaluate(tgt_encoder, src_classifier, val_loader)
        loss_trg, acc_trg, prob_trg, label_trg, _ = evaluate(tgt_encoder, src_classifier, tgt_test)

        if epoch % 5 == 0:
            print(
                f'Epoch[{epoch}],loss_val:{loss_val:.4f}||acc_val:{acc_val:.4f}'
                f'||loss_trg:{loss_trg:.4f}||acc_trg:{acc_trg:.4f}')
        if acc_val > acc_best:
            acc_best = acc_val
            trg_encoder_best = copy.deepcopy(tgt_encoder.state_dict())
            num_patience = 0
        else:
            num_patience += 1
        if num_patience > patience or acc_val == 1:
            break
        Loss_val.append(loss_val)
        Acc_val.append(acc_val)

    result_train = {'Loss_val': Loss_val, 'Acc_val': Acc_val}
    tgt_encoder.load_state_dict(trg_encoder_best)

    return tgt_encoder, acc_best, result_train


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def loss_disc(feat_src, feat_tgt, labels_src, tgt_pre, critic, criterion_disc, nclass):
    sample_src=feat_src[labels_src==nclass,:]
    sample_tgt=feat_tgt[tgt_pre==nclass,:]

    feat_concat = torch.cat((sample_src, sample_tgt), 0)

    pred_concat = critic(feat_concat.detach())

    label_src = (torch.ones(sample_src.size(0)).long()).cuda()
    label_tgt = (torch.zeros(sample_tgt.size(0)).long()).cuda()
    label_concat = torch.cat((label_src, label_tgt), 0)

    loss_critic = criterion_disc(pred_concat, label_concat)

    return loss_critic

def loss_adv(feat_tgt, soft_output, critic, criterion_tgt):
    entropy = Entropy(soft_output)  #
    entropy = 1.0 + torch.exp(-entropy)  #
    weight = entropy / torch.sum(entropy)  #

    # predict on discriminator
    pred_tgt = critic(feat_tgt)

    # prepare fake labels
    label_tgt = (torch.ones(feat_tgt.size(0)).long()).cuda()

    # compute loss for target encoder
    loss_tgt = torch.sum(criterion_tgt(pred_tgt, label_tgt) * weight.detach())

    return loss_tgt

