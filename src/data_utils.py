"""
Class for Data Loading and Pre-processing
"""

# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Classes
class Data:
    """
    Class to load data and perform several pre-processing steps
    """

    def __init__(self):
        """
        Initialize the class variables
        """

        self.logger = None                      # Logger object of the class
        self.src_fname = None                   # Filename (with path) of the dataset
        self.data = None                        # Data read from dataset
        self.feats = None                       # Features
        self.num_feats = None                   # Number of features / feature dimensions
        self.labels = None                      # Labels
        self.scaler = None                      # Type of feature scaling ('minmax' / 'std' / None)
        self.scaled_feats = None                # Scaled features
        self.train_data = {}                    # Dictionary of Train_X and Train_y
        self.test_data = {}                     # Dictionary of Test_X and Test_y
        self.initial_batch_size = None          # Intial batch size
        self.training_batch_size = None         # Subsequent batch size for training
        pass

    def set_logger(self, logger):
        """
        Sets logger object for the class
        :param logger: Logger object that contains logging configurations
        :return: None (Updates class variables)
        """

        self.logger = logger
        pass

    def load_csv(self, fname, header='infer'):
        """
        Load data from csv file
        :param fname: Filname (with path)
        :param header: Option to read header line from dataset file (int / list of int / None / 'infer')
        :return: Data read from the file (Also updates class variables)
        """

        self.src_fname = fname                                              # Set dataset filename
        self.logger.debug(f'Loading dataset from file: {fname}...')
        try:
            data_from_file = pd.read_csv(fname, header=header)              # Read from dataset
        except:
            self.logger.error(f'---> !!! Error in reading file: {fname} !!!')
        self.data = data_from_file.values                                   # Convert DataFrame to Numpy ndarray
        self.logger.debug(f'Loaded {self.data.shape[0]} samples from the file {fname.split("/")[-1]}')
        self.logger.info('Loading dataset done.')
        return self.data

    def get_feats_labels(self, data=None, label_column='last'):
        """
        Extract features and labels from data
        :param data: data from reading dataset (If None, select the class's data variable)
        :param label_column: location of the label column in the data
        :return: Features and Labels
        """

        self.logger.info('Extracting features and labels...')
        if data is None:                                                    # If external data is not given,
            data = self.data                                                # Choose class's data
        if label_column == 'last':                                          # If target label location is in the 'last'
            self.feats = data[:, :-1]                                       # Extract features
            self.labels = data[:, -1]                                       # Extract labels
            self.num_feats = self.feats.shape[1]                            # Calculate number of features
        elif label_column == 'first':                                       # If target label location is in the 'first'
            self.feats = data[:, 1:]                                        # Extract features
            self.labels = data[:, 1]                                        # Extract labels
            self.num_feats = self.feats.shape[1]                            # Calculate number of features
        else:
            self.logger.error(f'---> !!! Label column not valid !!!')
            raise ValueError

        self.logger.debug(f'Extracted {self.feats.shape[1]} features')
        self.logger.info('Extracting features and labels done.')
        return self.feats, self.labels

    def scale_features(self, feats=None, scale_type='minmax'):
        """
        Perform feature scaling
        :param feats: Unscaled features
        :param scale_type: Type of scaling ('minmax' / 'std' / None)
        :return: Scaled features
        """

        self.logger.info('Scaling features...')
        self.logger.debug(f'scaling type: {scale_type}')
        try:
            if feats is None:                                               # If featues is not given externally,
                feats = self.feats                                          # Select the class's features
            if scale_type == 'minmax':                                      # If scale type is minmax,
                self.logger.info('Performing minmax scaler')
                scaler = MinMaxScaler()
                self.scaled_feats = scaler.fit_transform(feats)             # Perform minmax scaling
            elif scale_type == 'std':                                       # If scale type is std,
                self.logger.info('Performing standard scaler')
                scaler = StandardScaler()
                self.scaled_feats = scaler.fit_transform(feats)             # Perform standard scaling
            elif scale_type == None:                                        # If scale type is None,
                self.logger.info('No scaling')
                self.scaled_feats = feats                                   # No scaling
            else:
                self.logger.warning('---> !!! Scaler type not valid. Scaling not performed. Returning original values !!!')
                self.scaled_feats = feats
        except:
            self.logger.error('---> !!! Error in scaling features !!!')
            raise ValueError

        self.logger.info('Scaling features done.')
        return self.scaled_feats

    def split_train_test(self, feats=None, labels=None, test_ratio=0.1):
        """
        Split the data (features and labels) into training and testing
        :param feats: Features
        :param labels: Labels
        :param test_ratio: Fraction of test samples to total samples
        :return: Train_X, Train_y, Test_X, Test_y
        """

        self.logger.info(f'Splitting into train and test data... Test ratio: {test_ratio}')
        if not (test_ratio > 0 and test_ratio < 1):
            self.logger.error(f'---> !!! Test ratio {test_ratio} is not valid !!!')     # Assert for validity of ratio
            raise AssertionError

        try:
            if feats is None:                                               # If features are not passed externally,
                feats = self.scaled_feats                                   # Choose class's scaled features
            if labels is None:                                              # If labels are not passed externally,
                labels = self.labels                                        # Choose class's labels

            test_size = int(len(feats) * test_ratio)                        # Calculate test size
            self.logger.info(f'Test size: {test_size}')
            train_X = feats[:-test_size]                                    # Extract train X
            train_y = labels[: -test_size]                                  # Extract train y
            test_X = feats[-test_size:]                                     # Extract test X
            test_y = labels[-test_size:]                                    # Extract test y

            self.logger.debug(f'train_X: {train_X.shape} train_y: {train_y.shape}'
                              f'test_X: {test_X.shape} test_y: {test_y.shape}')
            self.train_data = {'X': train_X, 'y': train_y}                  # Update train data
            self.test_data = {'X': test_X, 'y': test_y}                     # Update test data
            self.logger.info('Splitting into train and test data complete.')

        except:
            self.logger.error('---> !!! Error in Train Test Split !!!')
            raise ValueError

        return train_X, train_y, test_X, test_y

    def set_initial_batch_size(self, initial_batch_size):
        """
        Set initial batch/block size
        :param initial_batch_size: Size of the initial block
        :return: None (Updates class variables)
        """
        self.logger.debug(f'Setting initial batch size to {initial_batch_size}')
        self.initial_batch_size = initial_batch_size                        # Set initial_batch_size
        pass

    def set_training_batch_size(self, training_batch_size):
        """
        Set batch size for sequential training
        :param training_batch_size: Size of mini-batch used for training
        :return: None (Updates class variables)
        """

        self.logger.debug(f'Setting training batch size to {training_batch_size}')
        self.training_batch_size = training_batch_size                      # Set training_batch_size
        pass

    #Generators
    def fetch_train_data(self, initial_batch_size=None, training_batch_size=None, train_X=None, train_y=None):
        """
        Generator function to yield batch of data during training
        :param initial_batch_size: Size of initial batch of data
        :param training_batch_size: Size of mini-batch of data for sequential training
        :param train_X: Train X
        :param train_y: Train y
        :return: batch_id, X and y
        """

        if initial_batch_size is None:                                      # If initial batch size is not passed,
            initial_batch_size = self.initial_batch_size                    # Select class's initial batch size
        if training_batch_size is None:                                     # If training batch size is not passed,
            training_batch_size = self.training_batch_size                  # Select class's training batch size
        if train_X is None:                                                 # If train_X is not passed,
            train_X = self.train_data['X']                                  # Select class's train_X
        if train_y is None:                                                 # If train_y is not passed,
            train_y = self.train_data['y']                                  # Select class's train_y
        self.logger.debug(f'Fetching data. Initial batch size: {initial_batch_size},'
                          f'Training batch size: {training_batch_size}, '
                          f'Train_X: {train_X.shape}, Train_y: {train_y.shape}')

        batch_id = 0                                                        # Initialize batch_id to 0
        X = train_X[:initial_batch_size]                                    # Extract X
        y = train_y[:initial_batch_size]                                    # Extract y
        self.logger.debug(f'Yielding batch {batch_id} data, X: {X.shape}, y: {y.shape}')
        yield batch_id, X, y

        train_X = train_X[initial_batch_size:]
        train_y = train_y[initial_batch_size:]

        start = 0
        end = training_batch_size
        while end <= train_X.shape[0]:
            batch_id += 1                                                   # Increment batch id
            X = train_X[start:end, :]                                       # Extract X
            y = train_y[start:end]                                          # Extract y
            start += training_batch_size
            end += training_batch_size
            self.logger.debug(f'Yielding batch {batch_id} data, X: {X.shape}, y: {y.shape}')
            yield batch_id, X, y


# Main
if __name__ == '__main__':
    print('This module contains Data class that is required to load dataset and perform pre-processing steps')
    print('Dependency for: plt_mcc.py')
    print('Dependent on: pandas, scikit-learn')
    pass