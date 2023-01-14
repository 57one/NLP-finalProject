import numpy as np, argparse, time, pickle, random
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import DialogDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel, DialogueGCN_DailyModel, DialogueGCN_DialogModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
from functools import reduce
import math

seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5]
    # 4147, 4294, 1801, 303, 2804, 12413
    # labels_dict = {0: 12885, 1: 85572, 2: 1022, 3: 1150, 4: 174, 5: 1823, 6:353} # dailydialog
    labels_dict = {0: 4147, 1: 4294, 2: 1801, 3: 303, 4: 2804, 5: 12413}

    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights


def get_dialog_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = DialogDataset('train', path)
    testset = DialogDataset('test', path)
    validset = DialogDataset('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def eval_model(model, loss_function, dataloader):
    return 111

def write_to_excel(total_length, labels, preds):
    train = False
    if train:
        file1 = 'pred1-train.csv'
        file2 = 'pred2-train.csv'
    else:
        file1 = 'pred1-test.csv'
        file2 = 'pred2-test.csv'
        file3 = 'submit.csv'
    pd1, pd2, pd3 = pandas.DataFrame(), pandas.DataFrame(), pandas.DataFrame()
    sum = 0
    split_labels, split_preds = [], []
    last_labels, last_preds = [], []
    right_label = 0
    total_label = 0
    for i in total_length:
        split_labels.append(list(labels[sum : sum + i]))
        split_preds.append(list(preds[sum : sum + i]))
        sum += i
        last_labels.append(labels[sum - 1])
        last_preds.append(preds[sum - 1])
        if(labels[sum - 1] == preds[sum - 1]):
            right_label +=1
        total_label +=1

    split_labels = [int(''.join(str(x + 1) for x in l)) for l in split_labels]
    split_preds = [int(''.join(str(x + 1) for x in l)) for l in split_preds]
    # def fn(x, y):
    #     return (x + 1) * 10 + (y + 1) # error
    # split_labels = [reduce(fn, l) for l in split_labels]
    # split_preds = [reduce(fn, l) for l  in split_preds]
    last_labels = [i + 1 for i in last_labels]
    last_preds = [i + 1 for i in last_preds]
    id = [i + 1 for i in range(len(last_labels))]
    pd1['id'] = id
    pd1['split_labels'] = split_labels
    pd1['split_preds'] = split_preds

    pd2['id'] = id
    pd2['last_labels'] = last_labels
    pd2['last_preds'] = last_preds

    pd3['ID'] = id
    pd3['Last Label'] = last_preds
    # print(split_labels)
    # print(split_preds)
    # print(last_labels)
    # print(last_preds)
    pd1.to_csv(file1, index=False)
    pd2.to_csv(file2, index=False)
    pd3.to_csv(file3, index=False)
    print(right_label)
    print(total_label)
    print('accuracy:{}'.format(right_label / total_label))

    avg_accuracy = round(accuracy_score(last_labels, last_preds) * 100, 2)
    avg_fscore = round(f1_score(last_labels, last_preds, average='macro') * 100, 2)

    precision = round(precision_score(last_labels, last_preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    recall = round(recall_score(last_labels, last_preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    print('Acc:{}, macro-f1:{}, macro-recall:{}'.format(avg_accuracy, avg_fscore, recall))

def eval_graph_model(model, loss_function, dataloader, cuda):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    total_length = []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    model.eval()

    seed_everything()
    i = 0
    for data in dataloader:
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        total_length.extend(lengths)

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        i += 1
        if i == 50:
            break

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    write_to_excel(total_length, labels, preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)

    avg_fscore = round(f1_score(labels, preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    # Add precision and recall
    precision = round(precision_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    recall = round(recall_score(labels, preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, precision, recall

def eval():
    path = './saved/dialog/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='DialogRNN', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=15,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=15,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    n_classes = 6
    cuda = args.cuda
    batch_size = args.batch_size
    # change D_m into
    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    kernel_sizes = [3, 4, 5]
    weibo_pretrained = pickle.load(open('dialog/weibo_embedding_matrix_test', 'rb'))
    vocab_size, embedding_dim = weibo_pretrained.shape

    if args.graph_model:
        seed_everything()

        model = DialogueGCN_DialogModel(args.base_model,
                                       D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                       n_speakers=2,
                                       max_seq_len=110,
                                       window_past=args.windowp,
                                       window_future=args.windowf,
                                       vocab_size=vocab_size,
                                       n_classes=n_classes,
                                       listener_state=args.active_listener,
                                       context_attention=args.attention,
                                       dropout=args.dropout,
                                       nodal_attention=args.nodal_attention,
                                       no_cuda=args.no_cuda
                                       )
        model.init_pretrained_embeddings(weibo_pretrained)
        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'
    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print('Basic Dialog RNN Model.')
        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    name_list = [name, args.base_model, 'best']
    save_path = path + '_'.join(name_list) + '.pkl'
    model.load_state_dict(torch.load(save_path))

    if cuda:
        model.cuda()

    # weights = pickle.load(open('dialog/weight.pkl', 'rb'))
    # weights = [1 / x for x in weights]
    weights = create_class_weight()
    loss_weights = torch.FloatTensor(weights)

    if args.class_weight:
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    if args.class_weight:
        train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog_test.pkl',
                                                                     batch_size=batch_size, num_workers=0)
    else:
        train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog_test.pkl',
                                                                     batch_size=batch_size, num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    all_precision, all_recall = [], []

    # evaluate
    if args.graph_model:
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_precision, test_recall = eval_graph_model(
            model, loss_function, test_loader, cuda)
    else:
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_precision, test_recall = eval_model(model, loss_function, test_loader)

    print(
        'test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}'. \
            format(test_loss, test_acc, test_fscore, test_precision, test_recall))

if __name__ == '__main__':
    eval()

# if __name__ == '__main__':
#     path = './saved/dialog/'
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
#
#     parser.add_argument('--base-model', default='DialogRNN', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
#
#     parser.add_argument('--graph-model', action='store_true', default=True,
#                         help='whether to use graph model after recurrent encoding')
#
#     parser.add_argument('--nodal-attention', action='store_true', default=False,
#                         help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
#
#     parser.add_argument('--windowp', type=int, default=15,
#                         help='context window size for constructing edges in graph model for past utterances')
#
#     parser.add_argument('--windowf', type=int, default=15,
#                         help='context window size for constructing edges in graph model for future utterances')
#
#     parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
#
#     parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
#
#     parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
#
#     parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
#
#     parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
#
#     parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
#
#     args = parser.parse_args()
#     print(args)
#
#     args.cuda = torch.cuda.is_available() and not args.no_cuda
#     if args.cuda:
#         print('Running on GPU')
#     else:
#         print('Running on CPU')
#
#     n_classes = 6
#     cuda = args.cuda
#     batch_size = args.batch_size
#     # change D_m into
#     D_m = 100
#     D_g = 150
#     D_p = 150
#     D_e = 100
#     D_h = 100
#     D_a = 100
#     graph_h = 100
#     kernel_sizes = [3, 4, 5]
#     weibo_pretrained = pickle.load(open('dialog/weibo_embedding_matrix', 'rb'))
#     vocab_size, embedding_dim = weibo_pretrained.shape
#
#     if args.graph_model:
#         seed_everything()
#
#         model = DialogueGCN_DialogModel(args.base_model,
#                                        D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
#                                        n_speakers=2,
#                                        max_seq_len=110,
#                                        window_past=args.windowp,
#                                        window_future=args.windowf,
#                                        vocab_size=vocab_size,
#                                        n_classes=n_classes,
#                                        listener_state=args.active_listener,
#                                        context_attention=args.attention,
#                                        dropout=args.dropout,
#                                        nodal_attention=args.nodal_attention,
#                                        no_cuda=args.no_cuda
#                                        )
#         model.init_pretrained_embeddings(weibo_pretrained)
#         print('Graph NN with', args.base_model, 'as base model.')
#         name = 'Graph'
#     else:
#         if args.base_model == 'DialogRNN':
#             model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
#                                    n_classes=n_classes,
#                                    listener_state=args.active_listener,
#                                    context_attention=args.attention,
#                                    dropout_rec=args.rec_dropout,
#                                    dropout=args.dropout)
#
#             print('Basic Dialog RNN Model.')
#         elif args.base_model == 'GRU':
#             model = GRUModel(D_m, D_e, D_h,
#                              n_classes=n_classes,
#                              dropout=args.dropout)
#
#             print('Basic GRU Model.')
#
#
#         elif args.base_model == 'LSTM':
#             model = LSTMModel(D_m, D_e, D_h,
#                               n_classes=n_classes,
#                               dropout=args.dropout)
#
#             print('Basic LSTM Model.')
#
#         else:
#             print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
#             raise NotImplementedError
#
#         name = 'Base'
#
#     name_list = [name, args.base_model, 'best']
#     save_path = path + '_'.join(name_list) + '.pkl'
#     model.load_state_dict(torch.load(save_path))
#
#     if cuda:
#         model.cuda()
#
#     # weights = pickle.load(open('dialog/weight.pkl', 'rb'))
#     # weights = [1 / x for x in weights]
#     weights = create_class_weight()
#     loss_weights = torch.FloatTensor(weights)
#
#     if args.class_weight:
#         if args.graph_model:
#             loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
#         else:
#             loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
#     else:
#         if args.graph_model:
#             loss_function = nn.NLLLoss()
#         else:
#             loss_function = MaskedNLLLoss()
#
#     if args.class_weight:
#         train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog.pkl',
#                                                                      batch_size=batch_size, num_workers=0)
#     else:
#         train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog.pkl',
#                                                                      batch_size=batch_size, num_workers=0)
#
#     best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
#     all_fscore, all_acc, all_loss = [], [], []
#     all_precision, all_recall = [], []
#
#     # evaluate
#     if args.graph_model:
#         test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_precision, test_recall = eval_graph_model(
#             model, loss_function, test_loader)
#     else:
#         test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_precision, test_recall = eval_model(model, loss_function, test_loader)
#
#     print(
#         'test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}'. \
#             format(test_loss, test_acc, test_fscore, test_precision, test_recall))



