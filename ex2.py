# Moshe Samson <your name> 312297492 <your id>
import sys
lines_counter = 0


def create_output_file(filename):
    f = open(filename, "w")
    f.write("#Students	Moshe Samson    <your name>  312297492   <your id>" + '\n')


def write_to_output(line):
    global lines_counter

    f = open("output.txt", "a")
    f.write("#Output" + str(lines_counter) + '\t' + line + '\n')
    f.close()

    lines_counter += 1


def process_input_file_into_lines(filename):
    f = open(filename, 'r')
    f_lines = f.readlines()
    return_lines = [line for line in f_lines if (not(line.startswith('<')) and len(line)>2)]
    return return_lines


def get_all_events(f_lines):
    flat_items = [word for line in f_lines for word in line.split(" ")]
    return flat_items
    # d = {key: flat_items.count(key) for key in set_items}
    # return d


def main(args):
    total_events = 300,000
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
    write_to_output(str(total_events))

    f_lines = process_input_file_into_lines(devl_filename)
    events  = get_all_events(f_lines)
    # write_to_output()



if __name__ == "__main__":
    main(sys.argv)
