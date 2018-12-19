# Moshe Samson Roni Chernyak 312297492 312676091
import utils as ul
import math


class HeldOut:
    def __init__(self, dev_file, voc_size):
        '''
        Initializes Held Out model with given develop file and assumed vocabulary size
        :param dev_file: develop file to process
        :param voc_size: assumed vocabulary size
        '''
        voc, total_count, articles_content = ul.pre_process_set(dev_file)
        self.train_set, self.val_set, self.train_counter, self.val_counter, self.train_fr = ul.split_set(
            articles_content, 0.5, total_count)
        self.VOC_SIZE = voc_size

    def calc_ho_probability(self, word, fed_frequency=None):
        '''
        Calculates given words discounted probability.
        If fed_frequency parameter is passed, calculates the discounted probability of r = fed_frequency
        :param word: word for which probability will be calculated
        :param fed_frequency: if this parameter is set - calculates probability for r = fed_frequency
        :return: calculated probability
        '''
        if fed_frequency is None:
            w_fr = 0 if word not in self.train_counter else self.train_counter[word]
        else:
            w_fr = fed_frequency
        # words with w's frequency if it's 0: |V| - all the words we did encounter during training
        word_with_w_fr = self.VOC_SIZE - len(self.train_counter) if w_fr == 0 else len(self.train_fr[w_fr])
        nominator = self.calc_ho_nominator(w_fr)
        return nominator / (word_with_w_fr * len(self.val_set))

    def calc_ho_nominator(self, w_fr):
        '''
        Calculates t_r - nominator in the probability formula
        Calculates the sum of held out(validation) frequencies for all the words with w_fr frequency in training set
        :param w_fr: frequency to calculate
        :return: calculated nominator+972 54-532-4961
בנימין פרנקל
ב- output 7 יצא לי 415149 (אחד יותר ממה שיצא לכולם). למישהו יש רעיון למה זה קורה לי?
כתבתי למחרוזת רק שורות שלא מתחילות ב- <TRAIN
        '''
        # if w's frequency is 0: all the words we did not see during training and did see during val contribute to the sum
        return sum([self.val_counter.get(w, 0) for w in self.train_fr[w_fr]])


def check_proba_correctness(dev_file, input_word, voc_size):
    ho = HeldOut(dev_file, voc_size)
    unseen_proba = ho.calc_ho_probability(input_word, 0)
    # RT SANITY CHECK
    proba_sum = 0
    for w in ho.train_counter.keys():
        proba_sum += ho.calc_ho_probability(w)
    to_check = (unseen_proba * (ho.VOC_SIZE - len(ho.train_counter)) + proba_sum)
