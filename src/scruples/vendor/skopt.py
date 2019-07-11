# New BSD License
#
# Copyright (c) 2016 - scikit-optimize developers.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# a. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# b. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# c. Neither the name of the scikit-optimize developers nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Code vendored from skopt (scikit-optimize).

See https://github.com/scikit-optimize/scikit-optimize for the source.
"""

from skopt.utils import dump


# vendored from commit: 6740876a6f9ad92c732d394e8534a5236a8d3f84
class CheckpointSaver(object):
    """
    Save current state after each iteration with `skopt.dump`.


    Example usage:
        import skopt

        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])

    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint will be saved to;
    * `dump_options`: options to pass on to `skopt.dump`, like `compress=9`
    """
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        dump(res, self.checkpoint_path, **self.dump_options)
