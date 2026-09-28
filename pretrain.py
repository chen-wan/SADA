import copy
import itertools
import numpy as np
import torch
from torch import nn, optim
from evaluate import evaluate


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p)  #

    return optimizer


#
def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, val_loader, tgt_test, num_epochs, patience):

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()  #
    num_patience=0

    #
    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01, weight_decay=3e-4)  #

    #
    acc_best = 0
    Loss_train = []
    Loss_val = []
    Acc_val = []

    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        discriminator.train()
        loss_total = 0

        start_steps = epoch * len(source_train_loader)
        total_steps = num_epochs * len(source_train_loader)  #


        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, itertools.cycle(target_train_loader))):

            source_image, source_label = source_data
            target_image, target_label = target_data  #

            p = float(batch_idx + start_steps) / total_steps

            alpha = 2. / (1. + np.exp(-10 * p)) - 1  #

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()  #

            combined_image = torch.cat((source_image, target_image), 0)  #

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)  #
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)  #
            source_feature = encoder(source_image)  #

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)  #

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)  #

            domain_source_labels = torch.zeros(source_label.shape[0]).type('torch.LongTensor')  #
            domain_target_labels = torch.ones(target_label.shape[0]).type('torch.LongTensor')  #
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()  #
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)  #

            total_loss = class_loss + alpha*domain_loss  #
            total_loss.backward()  #
            optimizer.step()  #

            loss_total += total_loss.item()

        loss_train = loss_total/len(source_train_loader)

        loss_val, acc_val, prob_val, label_val, features_val = evaluate(encoder, classifier, val_loader)
        loss_test, acc_test, prob_test, label_test, _ = evaluate(encoder, classifier, tgt_test)

        if epoch % 5 == 0:
            print(
                f'Epoch[{epoch}],Loss_train:{loss_train:.4f}||Loss_val:{loss_val:.4f}'
                f'||acc_val:{acc_val:.4f}||acc_test:{acc_test:.4f}')
        if acc_val > acc_best:
            acc_best = acc_val
            encoder_best = copy.deepcopy(encoder.state_dict())
            classifier_best = copy.deepcopy(classifier.state_dict())
            num_patience = 0
        else:
            num_patience += 1
        if num_patience > patience or acc_val == 1:
            break
        Loss_train.append(loss_train)
        Loss_val.append(loss_val)
        Acc_val.append(acc_val)

    result_train = {'Loss_train': Loss_train, 'Loss_val': Loss_val, 'Acc_val': Acc_val}
    encoder.load_state_dict(encoder_best)
    classifier.load_state_dict(classifier_best)
    return encoder, classifier, acc_best, result_train













