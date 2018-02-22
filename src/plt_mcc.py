"""
Performs progressive learning multi-class classification
"""

# Imports
import os
from plt import PLT
from data_utils import Data


# Functions
def plt_multiclass(logger, **args):
    """
    Receives arguments from main.py and executes progressive multi-class ELM
    :param logger: Logger object that handles logging status
    :param args: command line arguments from main.py
    :return: None (Generates results file)
    """

    # CONFIGURE RESULTS FILE, DIRECTORY
    result_dir = '../results/'           # Results directory
    logger.info(f'Results directory: {result_dir}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)         # Create directory if doesn't exist
        logger.debug(f'Results directory does not exist. Creating the directory')
    log_fname = logger.handlers[0].baseFilename.split('/')[-1]
    result_fname = log_fname.replace('_log_', '_results_').replace('.log', '.results')      # Result filename
    logger.debug(f'Results filepath: {result_dir}{result_fname}')
    result_fh = open(f'{result_dir}{result_fname}', 'w')

    # BEGIN EXECUTION
    print('---PROGRESSIVE ELM - MULTI-CLASS---')
    print('---PROGRESSIVE ELM - MULTI-CLASS---', file=result_fh)

    # Read Arguments
    filename = args['filename']
    header = args['header']
    label_location = args['label']
    scale_type = args['scale']
    test_ratio = args['testratio']

    print(f'Dataset file: {filename}', file=result_fh)
    logger.debug(f'Dataset File: {filename}, Scaling Type: {scale_type}, Test ratio: {test_ratio}')
    logger.debug(f'Header attribute in pandas read_csv: {header}, label location in csv: {label_location}')

    # Hyper-parameters of the model
    hidden_dim = args['neurons']
    N0 = args['initial']
    batch_size = args['batch']

    logger.debug(f'Hidden layer neurons: {hidden_dim}, Number of samples in initial block: {N0}, '
                f'Batch size for training: {batch_size}')

    # Load and Pre-process Data
    logger.info('Loading and preprocessing data...')
    data = Data()               # Create an object of class Data to handle data operations
    data.set_logger(logger)     # Set logger for Data object
    print(f'Loading data from file: {filename}...')
    data.load_csv(fname=filename, header=header)        # Load dataset to data object
    data.get_feats_labels(label_column=label_location)  # Extract features and labels
    data.scale_features(scale_type=scale_type)          # Scale features
    data.split_train_test(test_ratio=test_ratio)        # Split to train, test
    logger.info('Loading and preprocessing data done')

    # PLT-ELM Model
    print('Creating PLT ELM model...')
    logger.info('Creating PLT ELM model')
    try:
        plt = PLT(data.num_feats, hidden_dim)           # Create an object of class PLT
    except:
        logger.error('---> !!! Error in creating PLT class object !!!')
        raise ValueError

    plt.set_logger(logger)                              # Set logger for plt object

    data.set_initial_batch_size(N0)                     # Set initial batch size
    data.set_training_batch_size(batch_size)            # Set training batch size

    # Training
    print('Begin Training...')
    logger.info('Begin Training...')
    for batch_id, X, y in data.fetch_train_data():      # Fetch data sequentially
        if batch_id == 0:
            plt.initial_batch(X, y)
            continue
        plt.train(X, y)                                 # Train model

    print('Training Complete')
    logger.info('Training Complete')

    # Testing
    print('Begin Testing...')
    logger.info('Begin Testing...')
    report, accuracy = plt.test(data.test_data['X'], data.test_data['y'])       # Test model
    print('Testing Complete')
    logger.info('Testing Complete')

    # Evaluation Metrics
    print(f'Classification Report: \n{report}\nAccuracy: {accuracy}')
    print(f'Classification Report: \n{report}\nAccuracy: {accuracy}', file=result_fh)
    logger.info(f'\nClassification Report: \n{report}')
    logger.info(f'\nAccuracy: {accuracy}')

    # Close results file handler
    result_fh.close()

    pass


# Main
if __name__ == '__main__':
    print('This file has function to perform progressive learning multi-class classifier')
    print('Dependency for: main.py')
    print('Dependent on: plt.py, data_utils.py')
    pass
