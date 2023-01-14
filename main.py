import numpy
import numpy as np, argparse, time, pickle, random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, DailyDialogueDataset2, DialogDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pickle
from torch.nn.utils.rnn import pad_sequence
import math

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]

    labels_dict = {0: 12885, 1: 85572, 2: 1022, 3: 1150, 4: 174, 5: 1823, 6: 353}

    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu * total / labels_dict[key])
        weights.append(score)
    print(weights)
    return weights

def append_label(path, output):
    test_csv = pd.read_csv(path, dtype=str, encoding='utf-8')
    for i in range(len(test_csv)):
        test_csv.at[i, 'Label'] = test_csv.at[i, 'Label'] + '1'
    test_csv.to_csv(output ,header = False, index = False, encoding='utf-8')


if __name__ == '__main__':
    # y_true = [0, 1, 2, 0]
    # y_pred = [0, 2, 1, 0]
    # print(f1_score(y_true, y_pred, average='macro'))
    append_label('./dialog/test_data_new(1).csv', './dialog/test_data_append(1).csv')
    # create_class_weight()
    # l = ['a', 'b', 'c']
    # print('-'.join(l))
    # l = [0, 1, 2, 3]
    # print(l[:1])
    # l1 = [1, 2, 3, 4]
    # l2 = [2, 2, 3, 3]
    # l = [1 if i == j else 0 for (i, j) in zip(l1, l2)]
    # print(l)
    # print(list(range(5)))
    # new_data = [[] for i in range(7)]
    # print(new_data)
    # a = torch.tensor([[1, 2], [2, 2], [3, 3], [3, 2], [3, 2]])
    # b = torch.tensor([[4, 2], [5, 1], [4, 4]])
    # c = torch.tensor([[6, 9]])
    # result = pad_sequence([a, b, c])
    # print(result)
    # print(result[0])
    # l = [1,2,3,4]
    # l = [1 / i for i in l]
    # print(l)
    # l = [1 / 0.0017,
    #      1 / 0.0034,
    #      1 / 0.1251,
    #      1 / 0.831,
    #      1 / 0.0099,
    #      1 / 0.0177,
    #      1 / 0.0112]
    # print(l)
    # pickle.dump(l, open('dialog/test.pkl', 'wb'))
    # ll = pickle.load(open('dialog/test.pkl', 'rb'))
    # print(ll)
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(int(True))
    # batch_size = 32
    # train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
    #                                                               batch_size=batch_size,
    #                                                               num_workers=0)
    # l = [0, 0, 0, 0, 0, 0, 0]
    # trainset = IEMOCAPDataset()
    # trainset = DialogDataset('train', 'dialog/dialog.pkl')
    # print(trainset.__getitem__(0)[0])
    # l = ['tr_c1', 'tr_c1', 'tr_c2', 'tr_c2', 'tr_c2', 'tr_c3', 'tr_c3', 'tr_c4', 'tr_c4',
    #  'tr_c2776']
    # l = list(dict.fromkeys(l))
    # l = list(dict.fromkeys(l))
    # print(l)
    # print(l en(trainset.__getitem__(0)[0]))
    # print(len(trainset.__getitem__(0)[5]))
    #
    # for i in range(len(trainset)):
    #     ll = trainset.__getitem__(0)[5].tolist()
    #     for i in ll:
    #         l[i] += 1
    # print(l)
    # print(numpy.sum(l))
    # print(l / numpy.sum(l))
    # print(len(trainset))
    # size = len(trainset)
    # idx = list(range(size))
    # print(idx)
    # print(idx[:])
    # train_sampler, valid_sampler = get_train_valid_sampler(trainset)
    # train_loader = DataLoader(trainset,
    #                           batch_size=32,
    #                           collate_fn=trainset.collate_fn,
    #                           num_workers=0,
    #                           pin_memory=False)
    # iter_loader = iter(train_loader)
    # print(next(iter_loader))
    # print(len(next(iter_loader)))
    # print(next(iter_loader)[6])
    # print(trainset.collate_fn(next(iter_loader)))
    # dat = pd.DataFrame(next(iter_loader))
    # k = 0
    # for i in dat:
    #     if k != 6:
    #         dat[i]
    #     else:
    #         print(dat[i])
    # trainset = DailyDialogueDataset2('train', 'dailydialog/daily_dialogue2.pkl')
    # print(trainset.__getitem__(0))
    # print(len(trainset.__getitem__(0)))
    # print(next(iter(train_loader))[0].tolist())
    # [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]
    # for i in trainset:
    #     print(i)
    # for i in trainset.__getitem__(0):
    #     print(i.size())
    # print(trainset.__getitem__(0)[5])

    # newL = [for i in range()]
    # l = [1, 2, 3, 4, 5]
    # q = [6, 7, 8, 9, 10]
    # dat = pd.DataFrame(l)
    # newL = [dat[i] for i in dat] # lazy calculate
    # print(newL)
    # l = [['Ses04F_script03_1', 'Ses04F_script02_2', 'Ses04F_impro02', 'Ses03F_impro06']]
    # dat = pd.DataFrame(l, columns=['idx'])
    # dat['qdx'] = q
    # print(dat)
    # for i, row in dat.iterrows():
    #     # print(row)
    #     print(row['idx'])
    # dict = { 'a':123,'b':456}
    # for k, v in dict.items():
    #     print(k,v)
    # for i in dict:
    #     print(i)
    # a = np.random.randn(3)
    # print(a)
    # a = np.zeros((1,2))
    # print(a.shape)
    # n_speakers = 2
    # edge_type_mapping = {}
    # for j in range(n_speakers):
    #     for k in range(n_speakers):
    #         edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
    #         edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
    # print(edge_type_mapping)

    # import torch
    # from ltp import LTP
    #
    # ltp = LTP("LTP/small")
    # if torch.cuda.is_available():
    #     # ltp.cuda()
    #     ltp.to("cuda")
    # output = ltp.pipeline(["他叫汤姆去拿外衣。", "武艺杰帅气"], tasks=["cws"])
    # print(output.cws)  # print(output[0]) / print(output['cws']) # 也可以使用下标访问
    # print(' '.join(output.cws))
    # dict = {'a':1}
    # set = {1,2,3}
    # print(set)

    # from transformers import pipeline, set_seed
    # generator = pipeline('text-generation', model = 'gpt2-medium')
    # set_seed(42)
    # l = generator("Hello, I'm a language model", max_length=30, num_return_sequences = 5)




