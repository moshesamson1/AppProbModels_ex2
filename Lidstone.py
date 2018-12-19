# Moshe Samson Roni Chernyak 312297492 312676091
import utils as ul
import random
import math


class Lidstone:
    def __init__(self, events, voc_size):
        """
        Initializes Lidstone model with given develop file and assumed vocabulary size
        :param events: all events from the develop file
        :param voc_size: assumed vocabulary size
        """
        train_size = int(0.9 * len(events))
        self.train_set = events[0:train_size]
        self.val_set = events[train_size:]
        self.voc_size = voc_size
        self.argmin_lamda = -1
        self.dic_train = ul.list_to_dictionary(self.train_set)
        self.training_set_size = len(self.train_set)

    def get_possible_events_amount(self):
        '''
        :return: assumed vocabulary size
        '''
        return self.voc_size

    def get_validation_set_size(self):
        '''
        :return: validation set size
        '''
        return len(self.val_set)

    def get_validation_set(self):
        '''
        :return: validation set
        '''
        return self.val_set

    def get_training_set_size(self):
        '''
        :return: training set size
        '''
        return self.training_set_size

    def get_training_set(self):
        '''
        :return: training set
        '''
        return self.train_set

    def get_mle_training(self, input_word):
        '''
        :return: frequency of given word in training set divided by training sets size
        '''
        return self.dic_train.get(input_word, 0) / self.get_training_set_size()

    def get_training_events(self):
        '''
        :return: all possible events in trining (unique words)
        '''
        return set(self.train_set)

    def get_argmin_lamda(self):
        '''
        :return: model's best lambda
        '''
        return self.argmin_lamda

    def calc_lid_probability_training(self, input_word, lamda, fr=None):
        '''
        Calculates given words discounted probability with given lambda.
        If fr parameter is passed, calculates the discounted probability of a word with given fr's occurrences
        :param input_word:  word for which probability will be calculated
        :param lamda:
        :param fr: if not not - ignores input word and calculates probability for a word with given fr's occurrences
        :return:
        '''
        w_count = self.dic_train.get(input_word, 0) if fr is None else fr
        return (w_count + lamda) / \
               (self.training_set_size + lamda * self.get_possible_events_amount())
