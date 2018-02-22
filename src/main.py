"""
PROGRESSIVE LEARNING - MULTI-CLASS CLASSIFICATION

Implementation of the paper title "Progressive Learning Technique for Multi-class Classification" published in Neurocomputing
Paper can be accessed via:
 Neurocomputing: https://www.sciencedirect.com/science/article/pii/S0925231216303137
 ArXiv: https://arxiv.org/pdf/1609.00085.pdf

Code developed in Python 3.6
Package Dependencies: numpy 1.14.0, pandas 0.21.1, scikit-learn 0.19.1
Custom Module Dependencies: plt_mcc(depends on plt, data_utils), log_utils
"""

# Imports
import argparse
from plt_mcc import plt_multiclass
from log_utils import Logger


# Functions
def parse_command_line_args():
    """
    Parses command line arguments to run progressive learning multi-class classifier using ELM
    List of command line arguments:
    -f FILENAME: Filename(with path) of dataset (default: '../datasets/iris_plt.csv')
    -hr HEADER: Option to read header line from dataset [header_row_no(int) / None / 'infer'](default: 'infer')
    -l LABEL_LOC: Position of target label column in the dataset ['first' / 'last'](default: 'last')
    -s SCALE: Type of feature scaling ['minmax' / 'std' / 'None'](default: 'minmax')
    -t TEST_RATIO: Fraction of test samples to total samples [0 to 1(float)](default: 0.1)
    -n NEURONS: Number of neurons in hidden layer [(int)](default: 10)
    -i INITIAL: Number of samples in initial block/batch [(int)](default: 30)
    -b BATCH: Number of samples in mini-batch training [(int)](default: 1)
    :return: key value paired arguments and values
    """

    parser = argparse.ArgumentParser(description='Progressive Multiclass ELM')
    parser.add_argument("-f", "--filename", default="../datasets/iris_plt.csv", help="Filename of the dataset (csv file)")
    parser.add_argument("-hr", "--header", default="infer", help="Header arguement for read_csv in pandas")
    parser.add_argument("-l", "--label", default="last", help="Location of label column in the csv file")
    parser.add_argument("-s", "--scale", default="minmax", help="Scaling type for feature scaling")
    parser.add_argument("-t", "--testratio", default=0.1, type=float, help="Ratio of test samples to total samples")
    parser.add_argument("-n", "--neurons", default=10, type=int, help="Number of neurons in hidden layer")
    parser.add_argument("-i", "--initial", default=30, type=int, help="Number of samples in initial block")
    parser.add_argument("-b", "--batch", default=1, type=int, help="Batch size for sequential training")
    args = parser.parse_args()
    return args


def main(args):
    """
    Setup logger and execute PLT classifier
    :param args: Arguments from the command line parsed using parse_command_line_args funtion
    :return: None
    """

    # LOGGER PARAMETERS
    LOG_FILE = 'plt_multiclass'
    # Root name of the log file. Execution time stamp will be appended to the filename
    # and will be created in ../logs/ directory
    LOG_LEVEL = 'info'
    # Log level: debug / info / warning / error / critical (Preferred: info)(Use debug during debugging)
    SHOW_LOG = False
    # If True, shows the log info in the run window.

    # LOGGER CONFIGURATION
    logger_config = Logger(__name__, log_file=LOG_FILE, log_level=LOG_LEVEL, show_log=SHOW_LOG)
    logger = logger_config.get_logger()

    # EXECUTE PLT MULTI-CLASS
    try:
        logger.info('Executing plt_multiclass with the arguments received')
        plt_multiclass(logger, **vars(args))
    except:
        print(f'\n\n!!! Error Occurred During Execution !!!\nCheck log file for further details\n'
              f'Log file: {logger.handlers[0].baseFilename}')
    pass


# Main
if __name__ == '__main__':
    args = parse_command_line_args()
    main(args)