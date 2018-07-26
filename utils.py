from common.input import observation_input

import numpy as np
import tensorflow as tf


def appending(destination, source_element):
    if type(source_element) == dict:
        if destination is None:
            destination = {key: list() for key in source_element}
        for key in source_element:
            destination[key].append(np.array(source_element[key], copy=False))
    else:
        if destination is None:
            destination = list()
        destination.append(np.array(source_element, copy=False))
    return destination

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(self, data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder, name):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        if type(data) != dict:
            return {self._placeholder: data}
        feed_dict = dict()
        for key in data:
            feed_dict[self._placeholder[key]] = data[key]
        return feed_dict


class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space
        
        Parameters
        ----------

        observation_space: 
                observation space of the environment. Should be one of the gym.spaces types
        name: str 
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt, name)

    def get(self):
        return self.processed_inpt
    
    
