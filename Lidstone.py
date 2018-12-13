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

    def get_validation_set_size(self):
        return len(self.val_set)

    def get_training_set_size(self):
        return len(self.train_set)

    def get_training_set(self):
        return self.train_set


