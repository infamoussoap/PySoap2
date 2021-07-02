import pandas as pd


class ModelLogger:
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None, auto_save=True):
        """ auto_save is used by Model - If true then the logger will save after training """
        self.model = model

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.auto_save = auto_save

        self.train_history = []
        self.test_history = []

    def log_model(self, epoch, batch_number=None):
        self.log_train_score(epoch, batch_number=batch_number)
        self.log_test_score(epoch, batch_number=batch_number)

    def log_train_score(self, epoch, batch_number=None):
        prediction = self.model.predict(self.x_train)
        loss = self.model._loss_function(prediction, self.y_train)
        metric = None if self.model._metric is None else self.model._metric(prediction, self.y_train)

        self.train_history.append(ModelEvalLog(epoch, batch_number, loss, metric))

    def log_test_score(self, epoch, batch_number=None):
        if self.x_test is None or self.y_test is None:  # Do nothing
            return None

        prediction = self.model.predict(self.x_test)
        loss = self.model._loss_function(prediction, self.y_test)
        metric = None if self.model._metric is None else self.model._metric(prediction, self.y_test)

        self.test_history.append(ModelEvalLog(epoch, batch_number, loss, metric))

    def save(self, train_filename=None, test_filename=None):
        if train_filename is None:
            train_filename = 'train_log.csv'
        if test_filename is None:
            test_filename = 'test_log.csv'

        self.train_history_as_df.to_csv(train_filename)
        self.test_history_as_df.to_csv(test_filename)

    @property
    def train_history_as_df(self):
        if len(self.train_history) == 0:
            return ModelEvalLog(None, None, None, None).as_dataframe()

        history = [log.as_dataframe() for log in self.train_history]
        return pd.concat(history)

    @property
    def test_history_as_df(self):
        if len(self.test_history) == 0:
            return ModelEvalLog(None, None, None, None).as_dataframe()

        history = [log.as_dataframe() for log in self.test_history]
        return pd.concat(history)


class Log:
    def __init__(self, **kwargs):
        self.log = kwargs

    def __str__(self):
        return str(self.log)


class ModelEvalLog(Log):
    def __init__(self, epoch, batch_number, loss, metric, **kwargs):
        super().__init__(loss=loss, metric=metric, **kwargs)

        self.epoch = epoch
        self.batch_number = batch_number

        self.loss = loss
        self.metric = metric

        self.other = kwargs

    def __str__(self):
        if self.batch_number is None:
            prefix = f'Epoch {self.epoch}: '
        else:
            prefix = f'Epoch {self.epoch}, Batch Number {self.batch_number}: '

        return prefix + super().__str__()

    def as_dataframe(self):
        df = pd.DataFrame([self.loss, self.metric, str(self.other)],
                          index=['Loss', 'Metric', 'Other'], columns=[self.epoch]).T

        df.index.name = 'Epoch'

        return df

    def __bool__(self):
        return True
