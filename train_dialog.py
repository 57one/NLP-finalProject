import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import DialogDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel, DialogueGCN_DailyModel, DialogueGCN_DialogModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
import math

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100

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

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


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


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []
    qmasks, umasks = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # import ipdb;ipdb.set_trace()
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        # print(lp_)
        # print(labels)
        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        # print(pred_)

        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        # print(masks)

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

        # print(masks)
        # print(labels)
        # print(preds)
        # print(vids)
        # break


    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)

    precision = round(precision_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    recall = round(recall_score(labels, preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)

    print(masks[:10])
    print(labels[:10])
    print(preds[:10])

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, vids, precision, recall


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    total_length = []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        total_length.extend(lengths)
        # print(total_length)
        # print(len(total_length))
        # print(umask)
        # print(len(umask))
        # print(umask[0])
        # print((umask[0] == 1))
        # print((umask[0] == 1).nonzero())
        # print((umask[0] == 1).nonzero().tolist())
        # print((umask[0] == 1).nonzero().tolist()[-1])
        # print((umask[0] == 1).nonzero().tolist()[-1][0])
        # print(lengths)
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

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

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

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)

    # avg_fscore = round(f1_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    # # Add precision and recall
    # precision = round(precision_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    # recall = round(recall_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)

    # for finalproject macro
    avg_fscore = round(f1_score(labels, preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    # Add precision and recall
    precision = round(precision_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)
    recall = round(recall_score(labels, preds, average='macro', labels=[0, 1, 2, 3, 4, 5]) * 100, 2)


    # print(len(total_length))
    # print(total_length[:5])
    print(labels[:10])
    print(preds[:10])
    sum = 0
    for i in total_length:
        sum += i
        # labels[sum-1] preds[sum-1]
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, precision, recall


if __name__ == '__main__':
    path = 'saved/dialog/'

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

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.6, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=90, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
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

    if cuda:
        model.cuda()

    # to be computed
    # weights = pickle.load(open('dialog/weight.pkl', 'rb'))
    weights = create_class_weight()
    # weights = [1 / x for x in weights]
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.class_weight:
        train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog_test.pkl',
                                                                     batch_size=batch_size, num_workers=0)
    else:
        train_loader, valid_loader, test_loader = get_dialog_loaders('dialog/dialog_test.pkl',
                                                                     batch_size=batch_size, num_workers=0)
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    all_precision, all_recall = [], []

    max_test_fscore = -1
    max_valid_fscore = -1
    best_epoch = 0
    for e in range(n_epochs):

        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, train_precision, train_recall = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, valid_precision, valid_recall = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda)
            # test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_precision, test_recall = train_or_eval_graph_model(
            #     model, loss_function, test_loader, e, cuda)
        else:
            # train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e,
            #                                                                       optimizer, True)
            # valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            # test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
            #                                                                                                      loss_function,
            #                                                                                                      test_loader,
            #                                                                                                      e)
            train_loss, train_acc, _, _, _, train_fscore, _, train_precision, train_recall = train_or_eval_model(model, loss_function, train_loader, e,
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _, valid_precision, valid_recall = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_precision, test_recall = train_or_eval_model(model,
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)

        all_fscore.append(valid_fscore)
        all_precision.append(valid_precision)
        all_recall.append(valid_recall)
        all_acc.append(valid_acc)

        name_list = [name, args.base_model, 'epoch-' + str(e + 1)]
        save_path = path + '_'.join(name_list) + '.pkl'
        torch.save(model.state_dict(), save_path)

        if valid_fscore > max_valid_fscore:
            best_epoch = str(e + 1)
            print('------' + str(e + 1) + '------')
            max_valid_fscore = valid_fscore
            name_list = [name, args.base_model, 'best']
            save_path = path + '_'.join(name_list) + '.pkl'
            torch.save(model.state_dict(), save_path)


        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, train_precision: {}, train_recall: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, valid_precision: {}, valid_recall: {},  time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, train_precision, train_recall, valid_loss,
                       valid_acc,
                       valid_fscore, valid_precision, valid_recall, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()
    print('Test performance..')
    index_max = all_fscore.index(max(all_fscore))
    print('F-Score:', all_fscore[index_max])
    print('Precision:', all_precision[index_max])
    print('Recall:', all_recall[index_max])
    print('Acc:', all_acc[index_max])
    print('best-epch:', best_epoch)

    #     all_fscore.append(test_fscore)
    #     all_precision.append(test_precision)
    #     all_recall.append(test_recall)
    #     all_acc.append(test_acc)
    #
    #     name_list = [name, args.base_model, 'epoch-' + str(e + 1)]
    #     save_path = path + '_'.join(name_list) + '.pkl'
    #     torch.save(model.state_dict(), save_path)
    #
    #     if test_fscore > max_test_fscore:
    #         best_epoch = str(e + 1)
    #         print('------' + str(e + 1) + '------')
    #         max_test_fscore = test_fscore
    #         name_list = [name, args.base_model, 'best']
    #         save_path = path + '_'.join(name_list) + '.pkl'
    #         torch.save(model.state_dict(), save_path)
    #
    #     if args.tensorboard:
    #         writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
    #         writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
    #
    #     print(
    #         'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, train_precision: {}, train_recall: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, valid_precision: {}, valid_recall: {}, test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}, time: {} sec'. \
    #             format(e + 1, train_loss, train_acc, train_fscore, train_precision, train_recall, valid_loss, valid_acc,
    #                    valid_fscore, valid_precision, valid_recall, test_loss, test_acc, test_fscore, test_precision,
    #                    test_recall, round(time.time() - start_time, 2)))
    #
    # if args.tensorboard:
    #     writer.close()
    # print('Test performance..')
    # index_max = all_fscore.index(max(all_fscore))
    # print('F-Score:', all_fscore[index_max])
    # print('Precision:', all_precision[index_max])
    # print('Recall:', all_recall[index_max])
    # print('Acc:', all_acc[index_max])
    # print('best-epch:', best_epoch)