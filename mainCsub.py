
import copy
import scipy.io as scio
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.model_selection import KFold
from model import*  #
from EEGdataLoad import*
from pretrain import*
from adapt import *

#
nameE = ['A01E','A02E','A03E','A04E','A05E','A06E','A07E','A08E','A09E']  #
nameT = ['A01T','A02T','A03T','A04T','A05T','A06T','A07T','A08T','A09T']  #
t = (2, 6)  #
patience = 50  #
epochs = 500  #
ratio_train = 0.5
ratio_val = 0.3

data = []
for sub_num in range(9):
    nameE_sub = nameE[sub_num]
    nameT_sub = nameT[sub_num]
    print(nameE_sub, nameT_sub)

    dataloaderE = EEGLoader2a2(nameE_sub, t)
    dataloaderT = EEGLoader2a2(nameT_sub, t)
    dataloader = ConcatDataset([dataloaderE, dataloaderT])  #
    data.append(dataloader)


kf = KFold(n_splits=9, shuffle=False) #
for src_index, tgt_index in kf.split(data):

    src_loader = []
    for i in src_index:
        dataS = data[i]
        src_loader = ConcatDataset([src_loader, dataS])

    tgt_loader = []
    for i in tgt_index:
        dataS = data[i]
        test_sub = i
        tgt_loader = ConcatDataset([tgt_loader, dataS])

    for cyc in range(10):

        len_train = int(ratio_train * len(tgt_loader))
        len_val = int(ratio_val * len(tgt_loader))
        len_test = len(tgt_loader) - len_val - len_train
        tgt_train, tgt_val, tgt_test = random_split(  #
            dataset=tgt_loader,
            lengths=[len_train, len_val, len_test],
            generator=torch.Generator().manual_seed(cyc*10),
        )

        src_train = DataLoader(dataset=src_loader, batch_size=32, shuffle=True) #
        tgt_val = DataLoader(dataset=tgt_val, batch_size=32, shuffle=True)
        tgt_train = DataLoader(dataset=tgt_train, batch_size=32, shuffle=True)
        tgt_test = DataLoader(dataset=tgt_test, batch_size=32, shuffle=True)

        src_encoder = TSNet(nChan=22, nClass=2).cuda()
        trg_encoder = TSNet(nChan=22, nClass=2).cuda()
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
        trg_encoder.load_state_dict(src_encoder.state_dict())
        trg_encoder, acc_best, result_train_tgt = train_trg(src_encoder, trg_encoder, discriminator, src_classifier,
                                src_train, tgt_train, tgt_val, tgt_test, patience, epochs)

        _, acc_train, prob_train, label_train, features_train = evaluate(src_encoder, src_classifier,
                                                                             src_train)
        loss_test, acc_test, prob_test, label_test, features_test = evaluate(trg_encoder, src_classifier, tgt_test)

        print(acc_test)
        path_base = './result/2a/'
        if not os.path.exists(path_base + 'S' + str(test_sub)):
            os.makedirs(path_base + 'S' + str(test_sub))
        path_res = path_base + 'S' + str(test_sub) + '/result' +str(cyc) + '.mat'
        scio.savemat(path_res, {'label_test': label_test, 'label_train': label_train,
                                'label_wo_test': label_wo_test, 'prob_wo_test': prob_wo_test,
                                'prob_test': prob_test, 'result_train_src': result_train_src,
                                'result_train_tgt': result_train_tgt})

        if cyc == 1:
            path_baseF = './resultF/2a/'
            if not os.path.exists(path_baseF + 'S' + str(test_sub)):
                os.makedirs(path_baseF + 'S' + str(test_sub))
            path_res = path_baseF + 'S' + str(test_sub) + '/result' +str(cyc) + '.mat'
            scio.savemat(path_res, {'features_wo_test': features_wo_test,
                                    'features_test': features_test,
                                    'features_train': features_train})


