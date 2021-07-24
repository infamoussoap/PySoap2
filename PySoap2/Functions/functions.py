from .ActivationFunctions import activation_functions
from .ErrorFunctions import error_functions
from .MetricFunctions import metric_functions


def get_activation_function(name, **kwargs):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    function = activation_functions.get(name, None)
    if function is None:
        raise Exception(f'{name} is not a defined function.')
    return function


def get_error_function(name):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    function = error_functions.get(name, None)
    if function is None:
        raise Exception(f'{name} is not a defined function.')
    return function


def get_metric_function(name):
    """ Returns the metric fucntion of a given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    function = metric_functions.get(name, None)
    if function is None:
        raise Exception(f'{name} is not a defined metric.')
    return function
