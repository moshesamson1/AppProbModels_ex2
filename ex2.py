# Moshe Samson <your name> 312297492 <your id>
import sys
import held_out as ho
import utils as ul

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
    return_lines = [line for line in f_lines if (not (line.startswith('<')) and len(line) > 2)]
    return return_lines


def get_all_events(f_lines):
    flat_items = [word for line in f_lines for word in line.split(" ")]
    return flat_items
    # d = {key: flat_items.count(key) for key in set_items}
    # return d


def handle_perplexity(ho_inst, lid, test_fiename):
    """
    outputs perplexity of each model and then outputs to file the best model
    :param ho_inst:
    :param lid:
    :param test_fiename:
    :return:
    """
    # TODO: FILL LIDSTONE PARTS
    # lid_perp = write_to_output(LID PERPLEXITY)
    ho_perp = write_to_output(str(ho_inst.calc_perplexity(test_fiename)))
    # write_to_output('L' if lid_perp > ho_perp else 'H')


def generate_table(f_ho, n_t_r, t_r):
    out = ""
    for r in range(10):
        # TODO: FILL LIDSTONE PARTS and use the first str
        #out += f"\n\t{r}\t{lid_array[r]}\t{f_ho[r]:.5f}\t{n_t_r[r]}\t{t_r[r]}"
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
    for r in range(10):
        f_ho.append(ho_inst.calc_ho_probability("", r))
        t_r.append(ho_inst.calc_ho_nominator(r))
        n_t_r.append(len(ho_inst.train_fr.get(r, 0)))

    # write outputs - should start from out 21
    write_to_output(str(len(ho_inst.train_set)))
    write_to_output(str(len(ho_inst.val_set)))
    write_to_output(str(ho_inst.calc_ho_probability(input_word)))
    write_to_output(str(ho_inst.calc_ho_probability("", 0)))
    return ho_inst, f_ho, n_t_r, t_r


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
    write_to_output(devl_filename)
    write_to_output(test_filename)
    write_to_output(input_word)
    write_to_output(output_filename)
    write_to_output(str(VOC_SIZE))

    f_lines = process_input_file_into_lines(devl_filename)
    events = get_all_events(f_lines)
    write_to_output('%f' % (events.count(input_word) / float(VOC_SIZE)))
    write_to_output(str(len(events)))

    # held out outs
    ho_inst, f_ho, n_t_r, t_r = handle_heldout(devl_filename, input_word, VOC_SIZE)

    # test output
    voc, total_count, articles_content = ul.pre_process_set(test_filename)
    write_to_output(str(total_count))

    # TODO: UNCOMMENT THIS AFTER LIDSTONE FILL
    # handle_perplexity(ho_inst,LID, test_filename)

    generate_table(f_ho, n_t_r, t_r)


if __name__ == "__main__":
    main(sys.argv)
