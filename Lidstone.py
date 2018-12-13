import utils as ul


class Lidstone:
    def __init__(self, dev_file, voc_size):
        '''
        Initializes Lidstone model with given develop file and assumed vocabulary size
        :param dev_file: develop file to process
        :param voc_size: assumed vocabulary size
        '''
        voc, total_count, articles_content = ul.pre_process_set(dev_file)
        self.train_set, self.val_set, self.train_counter, self.val_counter, self.train_fr = ul.split_set(
            articles_content, 0.9, total_count)
        self.VOC_SIZE = voc_size

    def get_validation_set_size(self):
        return len(self.val_set)

    def get_training_sett_size(self):
        return len(self.train_set)
