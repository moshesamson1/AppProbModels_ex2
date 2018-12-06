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


def main(args):
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


if __name__ == "__main__":
    main(sys.argv)
