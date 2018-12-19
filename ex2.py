# Moshe Samson <your name> 312297492 <your id>
import sys
import held_out as ho
import utils as ul
import math
from Lidstone import *
import re
import numpy as np

lines_counter = 1
VOC_SIZE = 300000
output_filename = ""


def create_output_file(filename):
    f = open(filename, "w")
    f.write("#Students	Moshe Samson    Roni Chernyak  312297492   312676091" + '\n')


def write_to_output(line):
    global lines_counter

    f = open(output_filename, "a")
    f.write("#Output" + str(lines_counter) + '\t' + line + '\n')
    f.close()

    lines_counter += 1


def process_input_file_into_lines(filename):
    f = open(filename, 'r')
    f_lines = f.readlines()
    # get every third row
    return_lines = f_lines[2::4]
    # return_lines = [line for line in f_lines if (not (line.startswith('<')) and len(line) > 2)]
    return return_lines


def get_all_events(f_lines):
    flat_items = [word for line in f_lines for word in line.split()]
    return flat_items


def handle_perplexity(test_voc, test_total_events, ho_inst, lid_inst):
    """
    outputs perplexity over test set of each model and then outputs to file the best model
    :param ho_inst:
    :param lid:
    :param test_fiename:
    :return:
    """
    lid_perp = calc_perplexity(test_voc, test_total_events,
                               lid_inst.calc_lid_probability_training,
                               lid_inst.get_argmin_lamda())
    ho_perp = calc_perplexity(test_voc, test_total_events, ho_inst.calc_ho_probability)
    write_to_output(str(lid_perp))
    write_to_output(str(ho_perp))
    write_to_output('L' if lid_perp > ho_perp else 'H')


def calc_perplexity(voc, total_count, proba_func, lamda=None):
    '''
    Calculates models perplexity over given vocabulary with given total count of events
    :param test_file:
    :return: calculated perplexity
    '''
    sum = 0
    words = voc.keys()
    for w in words:
        word_probability = proba_func(w) if lamda is None else proba_func(w, lamda)
        if word_probability > 0:
            sum += math.log(word_probability, 2) * voc[w]
    return 2 ** (-sum / total_count)


def generate_table(f_ho, n_t_r, t_r):
    out = ""
    for r in range(10):
        # TODO: FILL LIDSTONE PARTS and use the first str
        # out += f"\n\t{r}\t{lid_array[r]}\t{f_ho[r]:.5f}\t{n_t_r[r]}\t{t_r[r]}"
        out += "\n\t{0}\t{1}\t{2:.5f}\t{3}\t{4}".format(r, "LID", f_ho[r], n_t_r[r], t_r[r])
    write_to_output(out)


def handle_heldout(dev_file, input_word, voc_size):
    """
    outputs held out part outputs
    :param dev_file:
    :param input_word:
    :param voc_size:
    :return: returns instance of helf_out, array with fH for each r value, ntr array and tr array
    """
    ho_inst = ho.HeldOut(dev_file, voc_size)
    t_r = []
    n_t_r = []
    f_ho = []
    for r in range(0,10):
        f_ho.append(ho_inst.calc_ho_probability("", r)*len(ho_inst.train_set))
        t_r.append(ho_inst.calc_ho_nominator(r))
        n_t_r.append(len(ho_inst.train_fr.get(r, 0)))
    n_t_r[0] = VOC_SIZE - len(ho_inst.train_counter)
    # write outputs - should start from out 21
    write_to_output(str(len(ho_inst.train_set)))  # output 21
    write_to_output(str(len(ho_inst.val_set)))  # output 22
    write_to_output(str(ho_inst.calc_ho_probability(input_word)))  # output 23
    write_to_output(str(ho_inst.calc_ho_probability("unseen-word")))  # output 24
    return ho_inst, f_ho, n_t_r, t_r


def get_perplexity_lamda_argmin(devel_lid, dic, low, high, resolution=0.01):
    '''
    return arg min lamda of perplexity in range 0-2
    :param devel_lid: THe Lidstone model
    :param filename:
    :param low:
    :param high:
    :param resolution:
    :return:
    '''
    lamda_values = np.arange(low, high, resolution)
    results = [(
        calc_perplexity(dic, devel_lid.get_validation_set_size(), devel_lid.calc_lid_probability_training, lamda),
        lamda) for lamda in lamda_values]
    sort_by_perplexity = sorted(results, key=lambda tup: tup[0])
    return sort_by_perplexity[0][1]


def handle_Lidstone(devl_filename, input_word, VOC_SIZE):
    devel_f_lines = process_input_file_into_lines(devl_filename)
    devel_events = get_all_events(devel_f_lines)
    devel_lid = Lidstone(devel_events, VOC_SIZE)

    write_to_output(str(devel_lid.get_validation_set_size()))  # output 8
    write_to_output(str(devel_lid.get_training_set_size()))  # output 9
    write_to_output(str(len(devel_lid.get_training_events())))  # output 10
    write_to_output(str(devel_lid.get_training_set().count(input_word)))  # output 11
    write_to_output(str(devel_lid.get_mle_training(input_word)))  # output 12
    write_to_output(str(devel_lid.get_mle_training("unseen-word")))  # output 13
    write_to_output(str(devel_lid.calc_lid_probability_training(input_word, 0.10)))  # output 14
    write_to_output(str(devel_lid.calc_lid_probability_training("unseen-word", 0.10)))  # output 15

    dic = ul.list_to_dictionary(devel_lid.get_validation_set())
    write_to_output(
        str(calc_perplexity(dic, devel_lid.get_validation_set_size(), devel_lid.calc_lid_probability_training,
                            0.01)))

    write_to_output(
        str(calc_perplexity(dic, devel_lid.get_validation_set_size(), devel_lid.calc_lid_probability_training,
                            0.10)))
    write_to_output(
        str(calc_perplexity(dic, devel_lid.get_validation_set_size(), devel_lid.calc_lid_probability_training,
                            1.00)))
    argmin_lamda = get_perplexity_lamda_argmin(devel_lid, dic, 0.01, 2.00, 0.01)
    devel_lid.argmin_lamda = argmin_lamda
    write_to_output(str(argmin_lamda))  # output 19
    write_to_output(
        str(calc_perplexity(dic, devel_lid.get_validation_set_size(), devel_lid.calc_lid_probability_training,
                            argmin_lamda)))  # output 20
    return devel_lid


def main(args):
    global output_filename
    # total_events = "300000"
    if len(args) == 5:
        devl_filename = args[1]
        test_filename = args[2]
        input_word = args[3]
        output_filename = args[4]
    else:
        print("Not enough arguments! Exiting...")
        exit(-1)

    create_output_file(output_filename)
    write_to_output(devl_filename)  # output 1
    write_to_output(test_filename)  # output 2
    write_to_output(input_word)  # output 3
    write_to_output(output_filename)  # output 4
    write_to_output(str(VOC_SIZE))  # output 5

    f_lines = process_input_file_into_lines(devl_filename)
    events = get_all_events(f_lines)
    write_to_output(str(list(set(events)).count(input_word) / float(VOC_SIZE)))  # output 6
    write_to_output(str(len(events)))  # output 7

    # Lidstone model
    lid_model = handle_Lidstone(devl_filename, input_word, VOC_SIZE)

    # held out outs
    ho_inst, f_ho, n_t_r, t_r = handle_heldout(devl_filename, input_word, VOC_SIZE)
    #
    # # test output
    voc, total_count, articles_content = ul.pre_process_set(test_filename)
    write_to_output(str(total_count))
    handle_perplexity(voc, total_count, ho_inst, lid_model)
    #
    generate_table(f_ho, n_t_r, t_r)


if __name__ == "__main__":
    main(sys.argv)
