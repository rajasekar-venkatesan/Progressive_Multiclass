"""
PLT Class
"""

# Imports
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import accuracy_score, classification_report


# Classes
class PLT:
    """
    Class for Progressive Learning Technique
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initialization for class object
        :param input_dim: Number of neurons in input layer
        :param hidden_dim: Number of neurons in hidden layer
        Also sets various other class variables
        """

        self.input_dim = input_dim          # Number of input layer neurons
        self.hidden_dim = hidden_dim        # Number of hidden layer neurons
        self.output_dim = None              # Number of output layer neurons (Will be populated during initial batch)
        self.i2h = np.random.uniform(-1, 1, (input_dim, hidden_dim))        # Weights from input2hidden(i2h) layer
        self.h2o = None                     # Weights from hidden2output(h2o) layer (Will be populated during initial batch)
        self.labels_set = set()             # Set of all distinct labels that are exposed/known to the classifier
        self.labels2index_map = {}          # Mapping each label to an index
        self.internal_vars = {}             # Internal variables include H and M that are used for progressive learning
        self.activation = self._sigmoid     # Activation function
        self.logger = None                  # Logger of the class

    def set_logger(self, logger):
        """
        Set logger for class object
        :param logger: Logger object that contains logging configuration
        :return: None
        """

        self.logger = logger
        pass

    def initial_batch(self, X0, y0_actual):
        """
        Train initial batch for ELM
        (Training the initial batch for progressive ELM is slightly different from training of subsequent blocks)
        Objective: Configures number of output neurons and find the initial weights for hidden2output(h2o) layer
        :param X0: Features (m * n_f)[m: Number of samples, n_f: number of features]
        :param y0_actual: Labels (m * 1)
        :return: None (Updates class variables)
        """

        self.logger.debug(f'Processing Initial Batch')
        try:
            self._update_labels(set(y0_actual))         # Update labels exposed/known to the model
            y0 = self._label_to_bipolar(y0_actual)      # Convert raw label to index and to bipolar representation
            self._calculate_weights(X0, y0)             # Calculate model weights
        except:
            self.logger.error('---> !!! Error in processing initial batch !!!')
            raise ValueError
        self.logger.debug(f'Initial batch processed')
        pass

    def train(self, X, y_actual):
        """
        Train sequentially arriving mini-batches
        Objective: Update the weights of the model iteratively
        :param X: Features (b * n_f)[b: mini_batch size]
        :param y_actual: Labels (b * 1)
        :return: None (Updates class variables)
        """

        batch_labels = set(y_actual.tolist())           # Identify labels present in this batch
        new_labels = batch_labels - self.labels_set     # Find if any new labels are introduced
        if len(new_labels):                             # If new labels are found, update model architecture
            try:
                self.logger.debug('Updating model architecture...')
                self._update_model_architecture(new_labels)                     # Update model architecture
                self.logger.debug('Updating model architecture done.')
            except:
                self.logger.error('---> !!! Error in updating model architecture !!!')
                raise ValueError

        y = self._label_to_bipolar(y_actual)            # Convert raw labels to index and to bipolar representation
        self._update_weights(X, y)                      # Update model weights
        pass

    def predict(self, X):
        """
        Predict target labels from input features
        Objective: Find y_pred, given X
        :param X: Features (test_size * n_f)
        :return: y_pred: Predicted values of target labels
        """

        self.logger.debug('Predicting outputs...')
        try:
            out_hidden = self.activation(np.dot(X, self.i2h))                   # Calculate hidden layer activations
            y_pred = np.dot(out_hidden, self.h2o)                               # Calculate output layer activations
            y_pred = np.array([np.argmax(y_pred[i, :]) for i in range(len(y_pred))])    # Make prediction
        except:
            self.logger.error('---> !!! Error in making prediction !!!')
        self.logger.debug('Predicting outputs done.')
        return y_pred

    def get_label_indices(self, y):
        """
        Return indices of raw labels
        Objective: Find the indices of given labels using labels2index_map
        :param y: Raw labels
        :return: labels (Indices of raw labels)
        """

        self.logger.debug('Getting label indices')
        labels = np.array([self.labels2index_map[str(label)] for label in y])           # Retrieve indices of labels
        self.logger.debug('Returning label indices')
        return labels

    def test(self, X, y):
        """
        Perform testing of the model
        Objective: Using test X, predict y_pred and evaluate model by comparing with y
        :param X: Features (Test_X)
        :param y: Labels (Test_y)
        :return: report, accuracy (Classification report and accuracy score)
        """

        self.logger.debug('Testing...')
        y_pred = self.predict(X)                                                        # Make predictions for test X
        y_actual = self.get_label_indices(y)                                            # Convert raw labels to indices
        self.logger.debug('Testing done. Generating results')
        try:
            self.logger.debug('Generating classification report...')
            report = classification_report(y_actual, y_pred)                            # Calculate Classification Reprt
            self.logger.debug('Generating classification report done.')
        except:
            self.logger.error('---> !!! Error in generating classification report !!!')
            raise ValueError
        try:
            self.logger.debug('Calculating accuracy score...')
            accuracy = accuracy_score(y_actual, y_pred)                                 # Calculate Accuracy Score
            self.logger.debug('Calculating accuracy score done.')
        except:
            self.logger.error('---> !!! Error in calculating accuracy score !!!')
            raise ValueError
        return report, accuracy

    def _update_weights(self, X, y):
        """
        Updates the weights of the model
        Objective: Given X and y, update the weights to minimize the loss
        :param X: Features
        :param y: Labels
        :return: None (Updates model variables)
        """

        try:
            M = self.internal_vars['M']                                         # Internal variable
            beta = self.h2o                                                     # Read Hidden2output weights
            H = self.activation(np.dot(X, self.i2h))                            # Calculate hidden layer activation
            # Weight update steps
            Dr = np.eye(H.shape[0]) + np.dot(np.dot(H, M), np.transpose(H))
            Nr1 = np.dot(M, np.transpose(H))
            Nr2 = inv(Dr)
            Nr = np.dot(np.dot(np.dot(Nr1, Nr2), H), M)
            M = M - Nr
            Nr3 = y - np.dot(H, beta)
            Nr4 = np.dot(np.dot(M, np.transpose(H)), Nr3)
            beta = beta + Nr4
            self.h2o = beta                                                     # Update Hidden2output weights
            self.internal_vars['H'] = H                                         # Update internal variable H
            self.internal_vars['M'] = M                                         # Update internal variable M
        except:
            self.logger.error('---> !!! Error in updating weights !!!')
            raise ValueError
        pass

    def _update_model_architecture(self, new_labels):
        """
        Updates Model Architecture
        Objective: When new classes/labels are introduced, update model by increasing the number of output layer neurons
        and correspondingly modifying and updating the hidden2output weights
        :param new_labels: Newly introduced labels
        :return: None (Updates class variables)
        """

        self.logger.debug('Updating model architecture...')
        try:
            self._update_labels(new_labels)                                     # Update labels known to the model
            H = self.internal_vars['H']                                         # Read internal variable H
            M = self.internal_vars['M']                                         # Read internal variable M
            c = len(new_labels)                                                 # Number of newly introduced labels
            # Model update steps
            N_prime = self.hidden_dim
            m = len(self.labels_set) - len(new_labels)
            b = H.shape[0]  # Previous batch size
            beta = self.h2o

            beta_tilde = np.dot(beta, np.eye(m, m + c))
            Nr1 = np.dot(M, np.transpose(H))
            Nr2 = -1 * np.ones((b, c))
            delta_beta_c = np.dot(Nr1, Nr2)
            delta_beta_mc = np.concatenate((np.zeros((N_prime, m)), delta_beta_c), axis=1)
            beta_mc = beta_tilde + delta_beta_mc
            beta = beta_mc
            self.h2o = beta                                                     # Update model
        except:
            self.logger.error('---> !!! Error in updating model architecture !!!')
            raise ValueError
        self.logger.debug('Updating model architecture done.')

    def _calculate_weights(self, X0, y0):
        """
        Calculate weights during the initial batch/block
        :param X0: Features
        :param y0: Labels
        :return: None (Updates class variables)
        """

        try:
            H0 = self.activation(np.dot(X0, self.i2h))                          # Hidden layer activation
            # Calculate weight steps
            M0 = inv(np.dot(np.transpose(H0), H0))
            beta0 = np.dot(np.dot(M0, np.transpose(H0)), y0)
            H = H0
            M = M0
            self.h2o = beta0                                                    # Hidden2output weights
            self.internal_vars = {'H': H, 'M': M}                               # Set internal variables
        except:
            self.logger.error('---> !!! Error in calculating weights !!!')
            raise ValueError
        pass

    def _set_output_dim(self, output_dim):
        """
        Set the number of neurons in the output layer
        Being a progressive learning technique, number of output layer neurons changes when new classes are introduced
        :param output_dim: Number of neurons in the output layer
        :return: None (Updates class variables)
        """

        self.output_dim = output_dim                                            # Set output dimensions
        pass

    def _update_labels(self, new_labels):
        """
        Updates the set of labels and labels2index_map known to the model
        :param new_labels: Newly introduced classes/labels
        :return: None (Updates class variables)
        """

        self.logger.debug(f'Updating labels...')
        try:
            self.labels_set = self.labels_set | set(new_labels)                 # Update labels set
            for label in new_labels:
                self.labels2index_map[label] = len(self.labels2index_map)       # Update labels2index_map
            self._set_output_dim(len(self.labels2index_map))
        except:
            self.logger.error('---> !!! Error in updating labels !!!')
            raise ValueError
        self.logger.debug('Updating labels done.')
        pass

    def _label_to_bipolar(self, y):
        """
        Convert raw labels to indices and then, convert indices of labels to bipolar(+1 / -1) format
        :param y: Labels
        :return: bipolar representation of labels
        """

        try:
            y = np.array([self.labels2index_map[label] for label in y])         # Find indices
        except:
            self.logger.error('---> !!! Error in converting label to bipolar !!!')
            raise ValueError

        return self._to_bipolar(y)                                              # Convert to bipolar

    def _to_bipolar(self, y):
        """
        Converts indices of labels to bipolar (+1 / -1) format
        :param y: label indices
        :return: bipolar representation of labels
        """

        try:
            y_bipolar = np.ones((len(y), len(self.labels2index_map))) * -1      # Initialize all values to -1
            for i, label in enumerate(y):
                y_bipolar[i, label] = 1                                         # Update required values to +1
        except:
            self.logger.error('---> !!! Error in to_bipolar !!!')
            raise ValueError
        return y_bipolar

    def _sigmoid(self, data):
        """
        Sigmoid activation function
        :param data: input to the activation function
        :return: activated output
        """

        try:
            result = 1 / (1 + np.exp(-1 * data))                                # Sigmoid operation
        except:
            self.logger.error('---> !!! Error in calculating sigmoid !!!')
            raise ValueError
        return result


# Main
if __name__ == '__main__':
    print('This module contains PLT class that is required to perform progressive learning multi-class classification')
    print('Dependency for: plt_mcc.py')
    print('Dependent on: numpy, scikit-learn')
    pass
