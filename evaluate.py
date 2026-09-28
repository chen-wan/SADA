"""Test script to classify target data."""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def evaluate(encoder, classifier, test_loader):

    encoder.eval()
    classifier.eval()  #
    criterion = nn.CrossEntropyLoss()

    pre_label=[]
    label=[]
    prob=[]
    features=[]
    loss_total=0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()

            feature = encoder(data)
            output = classifier(feature)

            loss = criterion(output, target)

            loss_total += loss.item()
            _, predicted = torch.max(output, 1)
            pre_label.append(predicted.cpu())
            label.append(target.cpu())
            pre_prob = F.softmax(output, dim=1)
            prob.append(pre_prob.cpu())
            features.append(feature.cpu())


        avg_loss = loss_total/len(test_loader)

        label = torch.cat(label, dim=0)
        pre_label = torch.cat(pre_label, dim=0)
        prob = torch.cat(prob, dim=0)
        features = torch.cat(features, dim=0)
        accuracy = accuracy_score(label, pre_label)

    encoder.train()
    classifier.train()  #

    return avg_loss, accuracy, prob, label, features
