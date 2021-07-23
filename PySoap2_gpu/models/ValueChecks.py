import numpy as np
import pyopencl.array as cl_array

from PySoap2.utils import ImageAugmentationGenerator


def as_list_of_data_types(val, data_types, data_name):
    """ data_types assumed to be tuple of Object """
    if isinstance(val, data_types):
        return [val]
    elif all([isinstance(v, data_types) for v in val]):
        return val
    else:
        data_types_names = " or ".join([type_.__name__ for type_ in data_types])
        raise ValueError(f"{data_name} needs to be an instance of {data_types_names}, or a list of "
                         f"{data_types_names}")


def as_list_of_clarrays(device_queue, val, data_name, dtype=np.float64):
    if isinstance(val, cl_array.Array):
        if val.dtype == dtype:
            return [val]
        return [val.astype(dtype)]
    elif isinstance(val, np.ndarray):
        return [cl_array.to_device(device_queue, val.astype(dtype))]
    elif all([isinstance(v, (cl_array.Array, np.ndarray, ImageAugmentationGenerator)) for v in val]):
        return [convert_to_clarray(device_queue, v, dtype=dtype) for v in val]
    else:
        raise ValueError(f"Conversion of {data_name} to cl_array.Array is only supported for cl_array.Array, "
                         f"np.ndarray, or ImageAugmentationGenerator, or a list of them.")


def convert_to_clarray(device_queue, array, dtype=np.float64):
    """ Converts the array to cl_array.Array

        Parameters
        ----------
        device_queue : cl.CommandQueue
            The queue for the device to put the array on
        array : np.array or cl_array.Array
            The array to be converted
        dtype : Class, optional
            The data type for the array

        Notes
        -----
        If array is a cl_array.Array then it will be returned
        If array is np.array then it will be converted to the dtype and sent to the queue
    """
    if isinstance(array, cl_array.Array):
        if array.dtype == dtype:
            return array
        return array.astype(dtype)
    elif isinstance(array, np.ndarray):
        array = convert_to_contiguous_array(array)
        return cl_array.to_device(device_queue, array.astype(dtype))
    elif isinstance(array, ImageAugmentationGenerator):
        images = convert_to_contiguous_array(array.images)
        return cl_array.to_device(device_queue, images.astype(dtype))
    else:
        raise ValueError(f'{type(array)} not supported, only cl_array, np.ndarray and ImageAugmentationGenerator '
                         f'allowed to be converted to cl_array.Array.')


def convert_to_contiguous_array(array):
    if not is_array_contiguous(array):
        return np.ascontiguousarray(array)
    return array


def is_array_contiguous(array):
    """ array assumed to be np.array """
    return array.flags.forc
