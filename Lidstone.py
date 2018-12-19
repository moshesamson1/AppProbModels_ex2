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
        train_size = int(0.9*len(events))
        self.train_set = events[0:train_size]
        self.val_set = events[train_size:]
        self.voc_size = voc_size
        self.argmin_lamda = -1
        self.dic_train = ul.list_to_dictionary(self.train_set)
        self.training_set_size = len(self.train_set)

    def get_possible_events_amount(self):
        return self.voc_size

    def get_validation_set_size(self):
        return len(self.val_set)

    def get_validation_set(self):
        return self.val_set

    def get_training_set_size(self):
        return self.training_set_size

    def get_training_set(self):
        return self.train_set

    def get_mle_training(self, input_word):
        # compute mle
        return self.get_training_set().count(input_word) / float(self.get_training_set_size())

    def get_training_events(self):
        return set(self.train_set)

    def get_argmin_lamda(self):
        return self.argmin_lamda

    def calc_lid_probability_training(self, input_word, lamda):
        return float(self.dic_train.get(input_word,0) + lamda) / \
               float(self.training_set_size + lamda * self.get_possible_events_amount())