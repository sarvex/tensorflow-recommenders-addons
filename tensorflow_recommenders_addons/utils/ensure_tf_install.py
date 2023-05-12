# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Ensure the TensorFlow version is in the right range. This
# needs to happen before anything else, since the imports below will try to
# import TensorFlow, too.

from distutils.version import LooseVersion
import warnings

import tensorflow as tf

from tensorflow_recommenders_addons.version import MIN_TF_VERSION
from tensorflow_recommenders_addons.version import MAX_TF_VERSION


def _check_tf_version():
  """Warn the user if the version of TensorFlow used is not supported.

    This is not a check for custom ops compatibility. This check only ensure that
    we support this TensorFlow version if the user uses only Addons' Python code.
    """

  if "dev" in tf.__version__:
    warnings.warn(
        f"You are currently using a nightly version of TensorFlow ({tf.__version__}). \nTensorFlow Recommenders Addons offers no support for the nightly versions of TensorFlow. Some things might work, some other might not. \nIf you encounter a bug, do not file an issue on GitHub.",
        UserWarning,
    )
    return

  min_version = LooseVersion(MIN_TF_VERSION)
  max_version = LooseVersion(MAX_TF_VERSION)

  if min_version <= LooseVersion(tf.__version__) <= max_version:
    return

  warnings.warn(
      f"Tensorflow Recommenders Addons supports using Python ops for all Tensorflow versions above or equal to {MIN_TF_VERSION} and strictly below {MAX_TF_VERSION} (nightly versions are not supported). \n The versions of TensorFlow you are currently using is {tf.__version__} and is not supported. \nSome things might work, some things might not.\nIf you were to encounter a bug, do not file an issue.\nIf you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the Recommenders Addons's version. \nYou can find the compatibility matrix in Recommenders Addon's readme:\nhttps://github.com/tensorflow/recommenders-addons",
      UserWarning,
  )
