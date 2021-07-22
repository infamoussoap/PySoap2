import numpy as np
import pandas as pd

from .ValueChecks import as_list_of_data_type
from .ValueChecks import check_valid_targets_length


class ModelLogger:
    """ Logging module for model training

        Notes
        -----
        auto_save is used by the `model` attribute
            If true then the logger will save after training
            Otherwise, you will need to call the `save` method to save the log

        Attributes
        ----------
        model : :obj:Model
            Either the cpu Model or gpu Model
        x_train : np.array or cl_array.Array
        y_train : np.array or cl_array.Array
        x_test : np.array or cl_array.Array or None
        y_test : np.array or cl_array.Array or None
        auto_save : bool
             If set to true, then ModelLogger will automatically save once training is finished
             by the `model`
        train_history : list[ModelEvalLog]
            Saves the training history
        test_history : list[ModelEvalLog]
            Saves the testing history
    """
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None, auto_save=True):
        """ Initialise the logger with the model and train/test sets """
        self.model = model

        self.x_train = x_train
        self.y_train_as_list = as_list_of_data_type(y_train, np.ndarray, 'y_train')
        check_valid_targets_length(self.y_train_as_list, model.output_length, 'y_train')

        self.x_test = x_test
        if y_test is not None:
            self.y_test_as_list = as_list_of_data_type(y_test, np.ndarray, 'y_test')
            check_valid_targets_length(self.y_train_as_list, model.output_length, 'y_test')
        else:
            self.y_test_as_list = None

        self.auto_save = auto_save

        self.train_history = []
        self.test_history = []

    def log_model(self, epoch, batch_number=None):
        """ Log (or save) the current train and test score of the model

            Parameters
            ----------
            epoch : int
            batch_number : int, optional
        """
        self.log_train_score(epoch, batch_number=batch_number)
        self.log_test_score(epoch, batch_number=batch_number)

    def log_train_score(self, epoch, batch_number=None):
        """ Log the current training score of the model """
        predictions = self.model._predict_as_list(self.x_train)
        loss = self.model._loss_function_as_list(predictions, self.y_train_as_list)
        metric = self.model._metric_as_list(predictions, self.y_train_as_list)

        self.train_history.append(ModelEvalLog(epoch, batch_number, loss, metric))

    def log_test_score(self, epoch, batch_number=None):
        """ Log the current testing score of the model """
        if self.x_test is None or self.y_test_as_list is None:  # Do nothing
            return None

        predictions = self.model._predict_as_list(self.x_test)
        loss = self.model._loss_function_as_list(predictions, self.y_test_as_list)
        metric = self.model._metric(predictions, self.y_test_as_list)

        self.test_history.append(ModelEvalLog(epoch, batch_number, loss, metric))

    def save(self, train_filename=None, test_filename=None):
        """ Save the training and testing history as csv files

            Parameters
            ----------
            train_filename : str, optional
                The file name to save the training history
            test_filename : str, optional
                The file name to save the testing history
        """
        if train_filename is None:
            train_filename = 'train_log.csv'
        if test_filename is None:
            test_filename = 'test_log.csv'

        self.train_history_as_df.to_csv(train_filename)
        self.test_history_as_df.to_csv(test_filename)

    @property
    def train_history_as_df(self):
        """ Converts the attribute `train_history` to a pd.DataFrame.

            Notes
            -----
            The purpose of converting `train_history` to a dataframe is that
            it allows quicker saving of the training history
        """
        if len(self.train_history) == 0:
            return pd.DataFrame()

        history = pd.concat([log.as_dataframe() for log in self.train_history])
        history.columns = ['Total Loss'] + self.model.loss_functions + self.model.metric_functions + ['Other']

        non_none_columns = [col is not None for col in history.columns]
        return history.loc[:, non_none_columns]

    @property
    def test_history_as_df(self):
        """ Converts the attribute `test_history` to a pd.DataFrame.

            Notes
            -----
            The purpose of converting `test_history` to a dataframe is that
            it allows quicker saving of the testing history
        """
        if len(self.test_history) == 0:
            return pd.DataFrame()

        history = pd.concat([log.as_dataframe() for log in self.test_history])
        history.columns = ['Total Loss'] + self.model.loss_functions + self.model.metric_functions + ['Other']

        non_none_columns = [col is not None for col in history.columns]
        return history.loc[:, non_none_columns]


class Log:
    """ Base class for a logger

        Notes
        -----
        Initialising only accepts keyword arguments
    """
    def __init__(self, **kwargs):
        self.log = kwargs

    def __str__(self):
        return str(self.log)


class ModelEvalLog(Log):
    """ Logger for :obj:Model

        Attributes
        ----------
        epoch : int
            The current number of epochs the model has been trained
        batch_number : int or None
            The current batch number, within the epoch. If the batch number
            isn't being counted, then set this to None
        loss_vals : list[float] or None
            The loss on the dataset
        metrics : list[float] or none
            The metric on the dataset
        other : dict of str - :obj:
            Other keyword arguments, this should be used if there
            is something else that you want to be saved
    """
    def __init__(self, epoch, batch_number, loss_vals, metrics, **kwargs):
        """ Initialise the Log

            Notes
            -----
            Unlike :obj:Log, ModelEvalLog assumes to have the 4 base attributes
                epoch, batch_number, loss, metric

            If you want to save other attributes/features, then use keyword
            arguments to save it.
        """
        super().__init__(loss_vals=loss_vals, metrics=metrics, **kwargs)

        self.epoch = epoch
        self.batch_number = batch_number

        self.loss_vals = loss_vals
        self.total_loss = sum(loss_vals)
        self.metrics = metrics

        self.other = kwargs

    def __str__(self):
        if self.batch_number is None:
            prefix = f'Epoch {self.epoch}: '
        else:
            prefix = f'Epoch {self.epoch}, Batch Number {self.batch_number}: '

        return prefix + super().__str__()

    def as_dataframe(self):
        combined_loss_and_metric = [self.total_loss] + self.loss_vals + self.metrics
        df = pd.DataFrame(combined_loss_and_metric + [str(self.other)], columns=[self.epoch]).T
        df.index.name = 'Epoch'

        return df

    def __bool__(self):
        return True
