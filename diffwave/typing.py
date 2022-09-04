"""Custom types."""
from typing import Union

import numpy as np
import tensorflow as tf

TensorLike = Union[
    None, np.ndarray, tf.Tensor, tf.TensorShape, tf.TensorSpec, int, float
]
