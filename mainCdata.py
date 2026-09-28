
import copy
import scipy.io as scio
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.model_selection import KFold
from model import*  #
from EEGdataLoad import*
from pretrain import*
from adapt import *


patience = 50  #
epochs = 500  #
ratio_train = 0.5
ratio_val = 0.3

data2a_loader = loader_2a2((2,6))
data2b_loader = loader_2b((3,7))

src_loader = data2a_loader
tgt_loader = data2b_loader

#
for numK in range(30):

    len_train = int(ratio_train * len(tgt_loader))
    len_val = int(ratio_val * len(tgt_loader))
    len_test = len(tgt_loader) - len_val - len_train
    tgt_train, tgt_val, tgt_test = random_split(  #
        dataset=tgt_loader,
        lengths=[len_train, len_val, len_test],
        generator=torch.Generator().manual_seed(numK*10),
    )


    src_train = DataLoader(dataset=src_loader, batch_size=32, shuffle=True, drop_last=True) #
    tgt_val = DataLoader(dataset=tgt_val, batch_size=32, shuffle=True)
    tgt_train = DataLoader(dataset=tgt_train, batch_size=32, shuffle=True, drop_last=True)
    tgt_test = DataLoader(dataset=tgt_test, batch_size=32, shuffle=True)


    src_encoder = TSNet(nChan=3, nClass=2).cuda()
    trg_encoder = TSNet(nChan=3, nClass=2).cuda()  #
    src_classifier = Classifier(feature_dim=60, num_classes=2).cuda()
    discriminator = Discriminator(feature_dim=60, num_classes=2).cuda()
    discriminatorF = DiscriminatorF(feature_dim=60, num_classes=2).cuda()

    src_encoder, src_classifier, _, result_train_src = dann(src_encoder,
                                                            src_classifier,
                                                            discriminatorF,
                                                            src_train,
                                                            tgt_train,
                                                            tgt_val,
                                                            tgt_test,
                                                            epochs,
                                                            patience)

    _, acc_wo_test, prob_wo_test, label_wo_test, features_wo_test = evaluate(src_encoder, src_classifier,
                                                                             tgt_test)
    #
    trg_encoder.load_state_dict(src_encoder.state_dict())
    trg_encoder, acc_best, result_train_tgt = train_trg(src_encoder, trg_encoder, discriminator, src_classifier,
                                                        src_train, tgt_train, tgt_val, tgt_test, patience, epochs)

    _, acc_train, prob_train, label_train, features_train = evaluate(src_encoder, src_classifier,
                                                                     src_train)
    loss_test, acc_test, prob_test, label_test, features_test = evaluate(trg_encoder, src_classifier, tgt_test)


    print(acc_wo_test)
    print(acc_test)
    path_base = './result/ab/'
    if not os.path.exists(path_base):
        os.makedirs(path_base)
    path_res = path_base + '/result' + str(numK) + '.mat'
    scio.savemat(path_res, {'label_test': label_test, 'label_train': label_train,
                            'label_wo_test': label_wo_test, 'prob_wo_test': prob_wo_test,
                            'prob_test': prob_test, 'result_train_src': result_train_src,
                            'result_train_tgt': result_train_tgt})

    if numK==1:
        path_baseF = './resultF/ab/'
        if not os.path.exists(path_baseF):
            os.makedirs(path_baseF)
        path_res = path_baseF + '/result' + str(numK) + '.mat'
        scio.savemat(path_res, {'features_wo_test': features_wo_test,
                                'features_test': features_test,
                                'features_train': features_train})


