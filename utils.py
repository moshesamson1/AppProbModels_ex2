# Roni Chernyak 312676091
import math
import random


def pre_process_set(file):
    '''
    Initial preprocessing of given set file.
    :param file:
    :return: set's vocabulary, total number of events, all article's content array
    '''
    voc = dict()
    total_count = 0
    articles_content = []

    with open(file, "r") as f:
        content = f.readlines()
    # iterate over every 4th line - only content without headers and \n
    for line_idx in range(2, len(content), 4):
        articles_content.append(content[line_idx])
        words = content[line_idx].split()
        for w in words:
            voc[w] = voc.setdefault(w, 0) + 1
        total_count += len(words)
    return voc, total_count, articles_content


def add_encountered_words(words, words_dict):
    for w in words:
        words_dict[w] = words_dict.setdefault(w, 0) + 1


def split_set(content, split_amount, total_count):
    '''
    splits given content into train and validation sets
    :param content: content to split
    :param split_amount: values of split - should be in range (0,1]
    :param total_count: total number of events in given content
    :return: train set and validation set and their word-counting dictionaries
    '''
    assert split_amount > 1 or split_amount <= 0, "split_set parameter split_amount is out of range"
    train_size = math.round(split_amount * split_amount)
    # TODO: CHECK if val and train sets are needed as array or as numbers
    train_set, val_set = [], []
    train_counter, train_fr, val_counter = dict(), dict(), dict()

    # do random split
    shuffled_rows = random.shuffle(content)
    for line_idx in range(len(train_size)):
        words = shuffled_rows[line_idx].split()
        # if train set array reached desired value, add to validation array, otherwise add to train array
        if len(train_set) == train_size:
            val_set.extend(words)
            add_encountered_words(words, val_counter)
        else:
            train_set.extend(words)
            add_encountered_words(words, train_counter)
    assert len(train_set) != train_size or len(val_set) != total_count - train_size

    # create map of number of encounters -> words for train - ZERO FREQUENCIES should be handled in code
    for word, freq in train_set.items():
        if freq not in train_fr:
            train_fr[freq] = train_fr.setdefault(freq, []).append(word)

    return train_set, val_set, train_counter, val_counter, train_fr
