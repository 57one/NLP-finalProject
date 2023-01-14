import json
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import torch
from ltp import LTP

def split_excel(filename):
    train_csv_file = 'dialog/train.csv'
    valid_csv_file = 'dialog/valid.csv'
    # test_csv_file = 'dialog/test.csv'
    csvfile = open(filename, 'r', encoding='mbcs').readlines()
    lens = len(csvfile)
    train_csv = csvfile[1:int(lens * 0.9)]
    validation_csv = csvfile[int(lens * 0.9): ]
    # test_csv = csvfile[int(lens * 0.9) + 1:]
    with open(train_csv_file, 'w+', encoding='mbcs') as train_f, \
         open(valid_csv_file, 'w+', encoding='mbcs') as valid_f:
         # open(test_csv_file, 'w+', encoding='mbcs') as test_f:
        train_f.writelines(train_csv)
        valid_f.writelines(validation_csv)
        # test_f.writelines(test_csv)


def preprocess_text(x):
    for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punct, ' ')
    x = ' '.join(x.split())
    x = x.lower()
    return x


def split_dialogue(dialogue):
    dialogues = dialogue.split('__eou__')
    return dialogues


def create_utterances(filename, split):
    sentences,  emotion_labels, speakers, conv_id, utt_id = [], [], [], [], []

    lengths = []
    if split == 'test_append':
        train_csv = pd.read_csv(filename, names=['Text', 'Labels'], index_col=0, dtype=str, encoding='utf-8')
    else:
        train_csv = pd.read_csv(filename, names=['Text', 'Labels'], index_col = 0, dtype = str, encoding='mbcs')
    for c_id, (index, row) in enumerate(train_csv.iterrows()):
        dialogue = row['Text']
        labels = [c for c in row['Labels']]
        dialogues = split_dialogue(dialogue)
        for u_id, item in enumerate(zip(dialogues, labels)):
            sentences.append(item[0])
            emotion_labels.append(item[1])
            conv_id.append(split[:2] + '_c' + str(c_id))
            utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
            speakers.append(str(u_id % 2))

    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))
    data['emotion_label'] = [int(label) - 1 for label in emotion_labels]
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id
    return data


def load_pretrained_weibo():
    print("Loading weibo embedding model, this can take some time...")
    weibo_vector = {}
    f = open('H:\\57one\\word-embedding\\sgns.weibo.word', encoding='utf-8')
    f.readline()  # 195202 300

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            weibo_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading pretrained weibo embedding model.")
    return weibo_vector


def load_pretrained_glove():
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    # Put your glove embedding path here
    f = open('H:\\57one\\glove\\glove.840B.300d.txt', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading pretrained GloVe model.")
    return glv_vector


def encode_labels(encoder, l):
    return encoder[l]


def handle_test():
    # split_excel('dialog/train_data.csv')
    train_data = create_utterances('dialog/train.csv', 'train')
    valid_data = create_utterances('dialog/valid.csv', 'valid')
    test_data = create_utterances('dialog/test_data_append.csv', 'test_append')
    print(len(train_data), len(valid_data), len(test_data))

    # encode the emotion labels
    all_emotion_labels = set(train_data['emotion_label'])
    emotion_label_encoder, emotion_label_decoder = {}, {}
    for i, label in enumerate(all_emotion_labels):
        print(label)

    # count = [0] * len(all_emotion_labels)
    # for label in list(train_data['emotion_label']):
    #     count[label] += 1
    # print(count)
    # count = count / np.sum(count)
    # pickle.dump(count, open('dialog/weight.pkl', 'wb'))

    # split sentence
    ltp = LTP("LTP/small")
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")

    new_all_text = []
    all_text = list(train_data['sentence'])
    for text in all_text:
        output = ltp.pipeline(text, tasks=["cws"])
        new_all_text.append(' '.join(output.cws))
    train_data['sentence_split'] = new_all_text

    new_all_text = []
    all_text = list(valid_data['sentence'])
    for text in all_text:
        output = ltp.pipeline(text, tasks=["cws"])
        new_all_text.append(' '.join(output.cws))
    valid_data['sentence_split'] = new_all_text

    new_all_text = []
    all_text = list(test_data['sentence'])
    for text in all_text:
        output = ltp.pipeline(text, tasks=["cws"])
        new_all_text.append(' '.join(output.cws))
    test_data['sentence_split'] = new_all_text

    # tokenize all sentences

    all_text = list(train_data['sentence_split']) + list(valid_data['sentence_split']) + list(test_data['sentence_split'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)
    pickle.dump(tokenizer, open('dialog/tokenizer_test.pkl', 'wb'))

    # convert the sentences into sequences
    train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence_split']))
    valid_sequence = tokenizer.texts_to_sequences(list(valid_data['sentence_split']))
    test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence_split']))

    print(train_sequence[-10:])
    print(valid_sequence[-10:])
    print(test_sequence[-10:])

    train_data['sentence_length'] = [len(item) for item in train_sequence]
    valid_data['sentence_length'] = [len(item) for item in valid_sequence]
    test_data['sentence_length'] = [len(item) for item in test_sequence]

    max_num_tokens = 100

    train_sequence = pad_sequences(train_sequence, maxlen=max_num_tokens, padding='post')
    valid_sequence = pad_sequences(valid_sequence, maxlen=max_num_tokens, padding='post')
    test_sequence = pad_sequences(test_sequence, maxlen=max_num_tokens, padding='post')

    train_data['sequence'] = list(train_sequence)
    valid_data['sequence'] = list(valid_sequence)
    test_data['sequence'] = list(test_sequence)

    # save the data in pickle format
    convSpeakers, convInputSequence, convInputMaxSequenceLength, convEmotionLabels = {}, {}, {}, {}
    # train_conv_ids, test_conv_ids, valid_conv_ids = set(train_data['conv_id']), set(test_data['conv_id']), set(
    #     valid_data['conv_id'])
    train_conv_ids, test_conv_ids, valid_conv_ids = list(dict.fromkeys(train_data['conv_id'])), \
                                                    list(dict.fromkeys(test_data['conv_id'])), \
                                                    list(dict.fromkeys(valid_data['conv_id']))
    print(train_conv_ids[:10])
    print(valid_conv_ids[:10])
    print(test_conv_ids[:10])

    all_data = train_data.append(test_data, ignore_index=True).append(valid_data, ignore_index=True)

    print('Preparing dataset. Hang on...')
    for item in list(train_conv_ids) + list(test_conv_ids) + list(valid_conv_ids):
        df = all_data[all_data['conv_id'] == item]

        convSpeakers[item] = list(df['speaker'])
        convInputSequence[item] = list(df['sequence'])
        convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
        convEmotionLabels[item] = list(df['emotion_label'])

    pickle.dump([convSpeakers, convInputSequence, convInputMaxSequenceLength, convEmotionLabels,
                 train_conv_ids, test_conv_ids, valid_conv_ids], open('dialog/dialog_test.pkl', 'wb'))

    # save pretrained embedding matrix
    weibo_vector = load_pretrained_weibo()
    word_vector_length = len(weibo_vector['我'])
    word_index = tokenizer.word_index
    inv_word_index = {v: k for k, v in word_index.items()}
    num_unique_words = len(word_index)
    weibo_embedding_matrix = np.zeros((num_unique_words + 1, word_vector_length))

    for j in range(1, num_unique_words + 1):
        try:
            weibo_embedding_matrix[j] = weibo_vector[inv_word_index[j]]
        except KeyError:
            weibo_embedding_matrix[j] = np.random.randn(word_vector_length) / 200

    np.ndarray.dump(weibo_embedding_matrix, open('dialog/weibo_embedding_matrix_test', 'wb'))
    print('Done. Completed preprocessing.')

if __name__ == '__main__':
    # split_excel('dialog/train_data.csv')
    handle_test()

# if __name__ == '__main__':
#     # split_excel('dialog/train_data.csv')
#     train_data = create_utterances('dialog/train.csv', 'train')
#     valid_data = create_utterances('dialog/valid.csv', 'valid')
#     test_data = create_utterances('dialog/test.csv', 'test')
#     print(len(train_data), len(valid_data), len(test_data))
#
#     # encode the emotion labels
#     all_emotion_labels = set(train_data['emotion_label'])
#     emotion_label_encoder, emotion_label_decoder = {}, {}
#     for i, label in enumerate(all_emotion_labels):
#         print(label)
#
#     count = [0] * len(all_emotion_labels)
#     for label in list(train_data['emotion_label']):
#         count[label] += 1
#     print(count)
#     count = count / np.sum(count)
#     pickle.dump(count, open('dialog/weight.pkl', 'wb'))
#
#     # split sentence
#     ltp = LTP("LTP/small")
#     if torch.cuda.is_available():
#         # ltp.cuda()
#         ltp.to("cuda")
#
#     new_all_text = []
#     all_text = list(train_data['sentence'])
#     for text in all_text:
#         output = ltp.pipeline(text, tasks=["cws"])
#         new_all_text.append(' '.join(output.cws))
#     train_data['sentence_split'] = new_all_text
#
#     new_all_text = []
#     all_text = list(valid_data['sentence'])
#     for text in all_text:
#         output = ltp.pipeline(text, tasks=["cws"])
#         new_all_text.append(' '.join(output.cws))
#     valid_data['sentence_split'] = new_all_text
#
#     new_all_text = []
#     all_text = list(test_data['sentence'])
#     for text in all_text:
#         output = ltp.pipeline(text, tasks=["cws"])
#         new_all_text.append(' '.join(output.cws))
#     test_data['sentence_split'] = new_all_text
#
#     # tokenize all sentences
#
#     all_text = list(train_data['sentence_split']) + list(valid_data['sentence_split']) + list(test_data['sentence_split'])
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(all_text)
#     pickle.dump(tokenizer, open('dialog/tokenizer.pkl', 'wb'))
#
#     # convert the sentences into sequences
#     train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence_split']))
#     valid_sequence = tokenizer.texts_to_sequences(list(valid_data['sentence_split']))
#     test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence_split']))
#
#     print(train_sequence[-10:])
#     print(valid_sequence[-10:])
#     print(test_sequence[-10:])
#
#     train_data['sentence_length'] = [len(item) for item in train_sequence]
#     valid_data['sentence_length'] = [len(item) for item in valid_sequence]
#     test_data['sentence_length'] = [len(item) for item in test_sequence]
#
#     max_num_tokens = 100
#
#     train_sequence = pad_sequences(train_sequence, maxlen=max_num_tokens, padding='post')
#     valid_sequence = pad_sequences(valid_sequence, maxlen=max_num_tokens, padding='post')
#     test_sequence = pad_sequences(test_sequence, maxlen=max_num_tokens, padding='post')
#
#     train_data['sequence'] = list(train_sequence)
#     valid_data['sequence'] = list(valid_sequence)
#     test_data['sequence'] = list(test_sequence)
#
#     # save the data in pickle format
#     convSpeakers, convInputSequence, convInputMaxSequenceLength, convEmotionLabels = {}, {}, {}, {}
#     # train_conv_ids, test_conv_ids, valid_conv_ids = set(train_data['conv_id']), set(test_data['conv_id']), set(
#     #     valid_data['conv_id'])
#     train_conv_ids, test_conv_ids, valid_conv_ids = list(dict.fromkeys(train_data['conv_id'])), \
#                                                     list(dict.fromkeys(test_data['conv_id'])), \
#                                                     list(dict.fromkeys(valid_data['conv_id']))
#     print(train_conv_ids[:10])
#     print(valid_conv_ids[:10])
#     print(test_conv_ids[:10])
#
#     all_data = train_data.append(test_data, ignore_index=True).append(valid_data, ignore_index=True)
#
#     print('Preparing dataset. Hang on...')
#     for item in list(train_conv_ids) + list(test_conv_ids) + list(valid_conv_ids):
#         df = all_data[all_data['conv_id'] == item]
#
#         convSpeakers[item] = list(df['speaker'])
#         convInputSequence[item] = list(df['sequence'])
#         convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
#         convEmotionLabels[item] = list(df['emotion_label'])
#
#     pickle.dump([convSpeakers, convInputSequence, convInputMaxSequenceLength, convEmotionLabels,
#                  train_conv_ids, test_conv_ids, valid_conv_ids], open('dialog/dialog.pkl', 'wb'))
#
#     # save pretrained embedding matrix
#     weibo_vector = load_pretrained_weibo()
#     word_vector_length = len(weibo_vector['我'])
#     word_index = tokenizer.word_index
#     inv_word_index = {v: k for k, v in word_index.items()}
#     num_unique_words = len(word_index)
#     weibo_embedding_matrix = np.zeros((num_unique_words + 1, word_vector_length))
#
#     for j in range(1, num_unique_words + 1):
#         try:
#             weibo_embedding_matrix[j] = weibo_vector[inv_word_index[j]]
#         except KeyError:
#             weibo_embedding_matrix[j] = np.random.randn(word_vector_length) / 200
#
#     np.ndarray.dump(weibo_embedding_matrix, open('dialog/weibo_embedding_matrix', 'wb'))
#     print('Done. Completed preprocessing.')