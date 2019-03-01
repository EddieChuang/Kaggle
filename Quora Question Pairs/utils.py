from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import csv

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        emb_matrix = []
        word_to_index = {}
        index_to_word = {}
        for i, line in enumerate(f):
            line = line.strip().split()
            word = line[0]
            emb_matrix.append(np.array(line[1:], dtype=np.float64))
            index_to_word[i] = word
            word_to_index[word] = i
    
    return word_to_index, index_to_word, np.array(emb_matrix)


def read_train_data(train_file):
    tokenizer = RegexpTokenizer(r'\w+')
    with open(train_file, 'r', encoding='utf-8') as file:
        rows = csv.reader(file)
        next(rows)  # skip header

        sentence_pair = []
        is_duplicated = []
        sequence_length = []
        for row in rows:
            sent1 = tokenizer.tokenize(row[3].lower())
            sent2 = tokenizer.tokenize(row[4].lower())
            sentence_pair.append((sent1, sent2))
            is_duplicated.append(int(row[5]))
            sequence_length.append((len(sent1), len(sent2)))
    
    return np.array(sentence_pair), np.array(is_duplicated), np.array(sequence_length)


def read_test_data(test_file):
    tokenizer = RegexpTokenizer(r'\w+')
    with open(test_file, 'r', encoding='utf-8') as file:
        rows = csv.reader(file)
        next(rows)  # skip header

        sentence_pair = []
        sequence_length = []
        try:
            for i, row in enumerate(rows):
                sent1 = tokenizer.tokenize(row[1].lower())
                sent2 = tokenizer.tokenize(row[2].lower())
                sentence_pair.append((sent1, sent2))
                sequence_length.append((len(sent1), len(sent2)))

                if i % 100000 == 0:
                    print(i)

        except Exception as e:
            print(row)
    
    return np.array(sentence_pair), np.array(sequence_length)

def filter_unknown(sentence_pair, word_to_index, unk_char):
    unk_sent_pair = []
    for sent1, sent2 in sentence_pair:
        sent1 = [word if word in word_to_index else unk_char for word in sent1]
        sent2 = [word if word in word_to_index else unk_char for word in sent2]
        unk_sent_pair.append((sent1, sent2))
    
    return unk_sent_pair

def pad(sentence_pair, pad_char, max_len):
    padded_sent_pair = []
    for sent1, sent2 in sentence_pair:
        padded1 = sent1 + [pad_char] * (max_len - len(sent1))
        padded2 = sent2 + [pad_char] * (max_len - len(sent2))
        padded_sent_pair.append((padded1[:max_len], padded2[:max_len]))
    
    return padded_sent_pair

def word2index(sentence_pair, word_to_index):
    index_sent_pair = []
    for sent1, sent2 in sentence_pair:
        index_sent1 = [word_to_index[word] for word in sent1]
        index_sent2 = [word_to_index[word]  for word in sent2]
        index_sent_pair.append((index_sent1, index_sent2))
    
    return index_sent_pair

def index2word(index_sentence, index_to_word):
    return [index_to_word[index] for index in index_sentence]

def next_batch(x_train, y_train, sequence_length, batch_size):
    # sentenceA, sentenceB = x_train
    # seq_lenA, seq_lenB = sequence_length
    x_train = np.array(x_train)
    sequence_length = np.array(sequence_length)
    nbatch = len(x_train) // batch_size
    for i in range(nbatch):
        offset = i * batch_size
        batch_sentA = x_train[offset: offset + batch_size, 0]
        batch_sentB = x_train[offset: offset + batch_size, 1]
        batch_seq_lenA = sequence_length[offset: offset + batch_size, 0]
        batch_seq_lenB = sequence_length[offset: offset + batch_size, 1]
        duplicated = y_train[offset: offset + batch_size] if y_train.any() else []

        yield batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, duplicated
    
    offset = nbatch * batch_size
    batch_sentA = x_train[offset: , 0]
    batch_sentB = x_train[offset: , 1]
    batch_seq_lenA = sequence_length[offset: , 0]
    batch_seq_lenB = sequence_length[offset: , 1]
    duplicated = y_train[offset: ] if y_train.any() else []
    return batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, duplicated


def next_batch_with_pad(x_train, y_train, sequence_length, word_to_index, batch_size):
    def pad(sequence, max_len):
        return [seq + [word_to_index['<PAD>']] * (max_len-len(seq)) for seq in sequence]

    x_train = np.array(x_train)
    sequence_length = np.array(sequence_length)
    nbatch = len(x_train) // batch_size
    for i in range(nbatch):
        offset = i * batch_size
        batch_seq_lenA = sequence_length[offset: offset + batch_size, 0]
        batch_seq_lenB = sequence_length[offset: offset + batch_size, 1]
        batch_sentA = pad(x_train[offset: offset + batch_size, 0], max(batch_seq_lenA))
        batch_sentB = pad(x_train[offset: offset + batch_size, 1], max(batch_seq_lenB))
        duplicated = y_train[offset: offset + batch_size] if y_train.any() else []

        yield batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, duplicated
    
    offset = nbatch * batch_size
    batch_seq_lenA = sequence_length[offset: , 0]
    batch_seq_lenB = sequence_length[offset: , 1]
    
    batch_sentA = pad(x_train[offset: , 0], max(batch_seq_lenA))
    batch_sentB = pad(x_train[offset: , 1], max(batch_seq_lenB))
    duplicated = y_train[offset: ] if y_train.any() else []
    return batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, duplicated

def accuracy(probability, target):
    predict = np.round(probability)
    return np.mean(np.equal(predict, target))


def prediction2csv(prediction, filepath):
    with open(filepath, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_id', 'is_duplicate'])
        for i, pred in enumerate(prediction):
            writer.writerow([i, pred])


def split_train_val_data(X, Y, sequence_length, val_ratio):
    train_len = int(len(X) * (1 - val_ratio))
    x_train, y_train, train_sequlence_length = X[: train_len], Y[: train_len], sequence_length[: train_len]
    x_val, y_val, val_sequlence_length = X[train_len: ], Y[train_len: ], sequence_length[train_len: ]

    return x_train, y_train, train_sequlence_length, x_val, y_val, val_sequlence_length


def shuffle_data(X, Y, sequence_length):
    indice = np.arange(len(X))
    np.random.shuffle(indice)
    return X[indice, :], Y[indice], sequence_length[indice, :]


def word_match_feature(sent_pair, word_to_index):
    stop_words = set(stopwords.words('english'))
    stop_words = [word_to_index[w] for w in stop_words]

    wordA = set([w for w in sent_pair[0] if w not in stop_words])
    wordB = set([w for w in sent_pair[1] if w not in stop_words])

    matched_wordA = [w for w in wordA if w in wordB]
    matched_wordB = [w for w in wordB if w in wordA]

    return len(matched_wordA + matched_wordB) / len(wordA + wordB)

    
def tfidf_feature(sent_pair, weights, word_to_index):
    stop_words = set(stopwords.words('english'))
    stop_words = [word_to_index[w] for w in stop_words]

    wordA = set([w for w in sent_pair[0] if w not in stop_words])
    wordB = set([w for w in sent_pair[1] if w not in stop_words])

    shared_weights = [weights.get(w, 0) for w in wordA if w in wordB] + \
                     [weights.get(w, 0) for w in wordB if w in wordA]
    total_weights = [weights.get(w, 0) for w in wordA] + \
                    [weights.get(w, 0) for w in wordB]

    return np.sum(shared_weights) / np.sum(total_weights)
    


