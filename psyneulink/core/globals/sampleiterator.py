#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************   SAMPLER CLASSES ********************************************************
"""

* `SampleSpec`
* `SampleIterator`

"""

import numpy as np

import typecheck as tc
from collections.abc import Iterator
from inspect import isclass
from decimal import Decimal, getcontext
from numbers import Number


__all__ = ['SampleSpec', 'SampleIterator']


# Number of decimal places used to set precision of decimal module
SAMPLE_SPEC_PRECISION = 16


def _validate_function(source, function):
    """Ensure function specification is appropriate for SampleIterator"""
    source_name = source.__class__.__name__
    try:
        result = function()
    except:
        raise SampleIteratorError("Specification of function for {} ({}) could not be called as a function".
                                  format(source_name, function.__class__.__name__))
    if result is None:
        raise SampleIteratorError("Function specified for {} ({}) does not return a result)".
                                  format(source_name, repr(function)))
    if not isinstance(result, Number):
        raise SampleIteratorError("Function specified for {} ({}) does not return a number)".
                                  format(source_name, repr(function)))


class SampleIteratorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class SampleSpec():
    """
    SampleSpec(      \
    start=None,      \
    stop=None,       \
    step=None,       \
    num=None,        \
    function=None    \
    precision=None   \
    custom_spec=None \
    )

    Specify the information needed to create a SampleIterator that will either (1) generate discrete values in a range;
    (2) call a function can generate continous values within the range; (3) pass a custom spec to SampleIterator

    (1) Generate values in a range by explicitly specifying a finite regular sequence of values, using an appropriate
        combination of the **start**, **stop**, **step** and/or **num**arguments.

        * if **start**, **stop**, and **step** are specified, the behavior is similar to `Numpy's arange
          <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.arange.html>`_. Calling
          `next <SampleIterator.__next__>`first returns **start**. Each subsequent call to next returns
          **start** :math:`+` **step** :math:`*` current_step.
          Iteration stops when the current value exceeds the **stop** value.  If any of the specified arguments are
          floats, precision determines the number of decimal places used for rounding to ensure num is a int.

        * if **start**, **stop**, and **num** are specified, the behavior is similar to `Numpy's linspace
          <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linspace.html>`_. Calling
          `next <SampleIterator.__next__>` first returns **start**. Each subsequent call to next returns
          **start** :math:`+` step :math:`*` current_step, where step is set to :math:`\\frac{stop-start)}{num - 1}`.
          Iteration stops when **num** is reached, or the current value exceeds the **stop** value.

        * if **start**, **stop*, **step**, and **num** are all specified, then **step** and **num** must be compatible.

    (2) Specify a function, that is called repeatedly to generate a sequence of values.

        * if **num** is specified, the **function** is called once on each iteration until **num** iterations are
          complete.

        * if **num** is not specified, the **function** is called once on each iteration, and iteration may continue
          indefintely.

    (3) Specify a custom_spec, that is passed to SampleIterator unmodified.

        * this is useful if the receiver of the SampleIterator uses a custom format or external functions to generate
          samples;  the format must, of course, be one expected by the receiver.

    .. note::
        * Some OptimizationFunctions may require that their SampleIterators have a "num" attribute.
        * The Python decimal module is used to implement **precision** for rounding.


    Arguments
    ---------

    start : int or float
        first sample in the sequence.

    stop : int or float
        maximum value of last sample in the sequence.

    step : int or float
        space between samples in sequence.

    num :  int
        number of samples.

    function : function or function class
        function to be called on each iteration. Must return one sample each time it is called.

    precision : int default 16
        number of decimal places used for rounding in floating point operations.

    custom_spec : anything

    Attributes
    ----------

    start : float
        first sample in the sequence.

    stop : float
        maximum value of last sample in the sequence.

    step : float
        space between samples in sequence.

    num :  int
        number of samples.

    function : function
        Function to be called on each iteration. Must return one sample.

    custom_spec : anything
        specification in a format recognized by receiver of SampleIterator.

    """

    @tc.typecheck
    def __init__(self,
                 start:tc.optional(tc.any(int, float))=None,
                 stop:tc.optional(tc.any(int, float))=None,
                 step:tc.optional(tc.any(int, float))=None,
                 num:tc.optional(int)=None,
                 function:tc.optional(callable)=None,
                 precision:tc.optional(int)=None,
                 custom_spec = None
                 ):

        self.custom_spec = custom_spec

        if self.custom_spec:
            # Assumes receiver of SampleIterator will get this and know what to do with it,
            #   therefore no other attributes are needed and, to avoid confusion, they should not be available;
            #   so just assign and return.
            return

        self._precision = precision or SAMPLE_SPEC_PRECISION
        # Save global precision for later restoration
        _global_precision = getcontext().prec
        # Set SampleSpec precision
        getcontext().prec = self._precision

        if function is None:
            if start is None or stop is None:
                raise SampleIteratorError("If 'function' is not specified, then 'start' and 'stop' must be "
                                                "specified.")
            if num is None and step is not None:
                num = int(Decimal(1.0) + Decimal(stop - start) / Decimal(step))
                # num = 1.0 + (stop - start) / step
            elif step is None and num is not None:
                step = (Decimal(stop) - Decimal(start)) / (num - 1)
            elif num is None and step is None:
                raise SampleIteratorError("Must specify one of {}, {} or {}."
                                                .format(repr('step'), repr('num'), repr('function')))
            else:
                if not np.isclose(num, 1.0 + (stop - start) / step):
                    raise SampleIteratorError("The {} ({}) and {} ({}} values specified are not comaptible."
                                                    .format(repr('step'), step, repr('num'), num))

        elif callable(function):
            _validate_function(self, function)

            if start is not None:
                raise SampleIteratorError("Only one of {} ({}) and {} ({}} may be specified."
                                                .format(repr('start'), start, repr('function'), function))
            if step is not None:
                raise SampleIteratorError("Only one of {} ({}) and {} ({}} may be specified."
                                                .format(repr('step'), step, repr('function'), function))
        else:
            raise SampleIteratorError("{} is not a valid function for {}."
                                            .format(function, self.__name__))

        # FIX: ELIMINATE WHEN UPGRADING TO PYTHON 3.5.2 OR 3.6, (AND USING ONE OF THE TYPE VERSIONS COMMENTED OUT ABOVE)
        # Validate entries of specification
        #
        self.start = start
        self.stop = stop
        self.step = step
        self.num = num
        self.function = function

        # Restore global precision
        getcontext().prec = _global_precision


allowable_specs = (tuple, list, np.array, range, np.arange, callable, SampleSpec)
def is_sample_spec(spec):
    if spec is None or type(spec) in allowable_specs:
        return True
    return False


class SampleIterator(Iterator):
    """
    SampleIterator(               \
    specification                 \
    )

    Create an iterator that returns the next sample from a sequence on each call to `next <SampleIterator.__next__>`.
    (e.g., when next(<SampleIterator>) is called)

    The pattern of the sequence depends on the **specification**, which may be a list, nparray, range,
    function, or a SampleSpec. Most of the patterns depend on the "current_step," which is incremented on each
    iteration, and set to zero when the iterator is reset.

    +--------------------------------+-------------------------------------------+------------------------------------+
    | **Specification**              |  **what happens on each iteration**       | **StopIteration condition**        |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | list, nparray                  | look up the item with index current_step  | list/array                         |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | range, np.arange               | start + step*current_step                 | range stop value is reached        |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | callable                       | call callable                             | iteration does not stop            |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | SampleSpec(start, stop, step)  | start + step*current_step                 | current_step = num or value > stop |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | SampleSpec(start, stop, num)   | start + step*current_step                 | current_step = num or value > stop |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | SampleSpec(function, num)      | call function                             | current_step = num                 |
    +--------------------------------+-------------------------------------------+------------------------------------+
    | SampleSpec(function)           | call function                             | iteration does not stop            |
    +--------------------------------+-------------------------------------------+------------------------------------+

    .. note::
        We recommend reserving the list/nparray option for cases in which the samples do not have a pattern that can be
        represented by a SampleSpec, or the number of samples is small. The list/nparray option requires all of the
        samples to be stored and looked up, while the SampleSpec options generate samples as needed.

    """

    @tc.typecheck
    def __init__(self,
                 specification:tc.any(*allowable_specs)):
        """

        Arguments
        ---------

        specification : list, array, range, callable or SampleSpec
            specifies what to use for `generate_current_value <SampleIterator.generate_current_value>`.

        Attributes
        ----------

        start : scalar or None
            first sample in the sequence, or None

        stop : scalar or None
            last sample in the sequence, or None

        step : scalar
            space between samples in the sequence, or None

        num : int or None
            number of values returned before the iterator stops. If None, the iterator may be called indefinitely.

        current_step : int
            number of current iteration

        specification :  list, np.array, range, np.arange, callable, SampleSpec
            original argument provided in constructor;
            useful for passing application-specific forms of specification directly to receiver of SampleIterator.

        Returns
        -------

        List(self) : list
        """

        # FIX: DEAL WITH head?? OR SIMPLY USE CURRENT_STEP?
        # FIX Are nparrays allowed? Below assumes one list dimension. How to handle nested arrays/lists?
        self.specification = specification

        if isinstance(specification, (tuple, range)):
            specification = list(specification)

        elif callable(specification):
            if isclass(specification):
                specification = specification()
            _validate_function(self, specification)

            specification = SampleSpec(function=specification)

        if isinstance(specification, list):
            self.start = specification[0]
            self.stop = None
            self.step = None
            self.num = len(specification)
            self.generator = specification                       # the list

            def generate_current_value():                        # index into the list
                # KDM 12/11/19: for currently unknown and unreplicable
                # reasons, the checks in __next__ will fail to ensure
                # that self.current_step is less than the length of
                # self.generator, and an IndexError will be thrown
                # occurs in tests/log/test_log.py::TestFiltering
                # and tests/models/test_greedy_agent.py due to
                # GridSearch use
                try:
                    return self.generator[self.current_step]
                except IndexError:
                    raise StopIteration

        elif isinstance(specification, SampleSpec):

            if specification.custom_spec:
                # Assumes receiver of SampleIterator will get this and know what to do with it,
                #   therefore no other attributes are needed and, to avoid confusion, they should not be available;
                #   so just return.
                return

            if specification.function is None:
                self.start = specification.start
                self.stop = specification.stop
                # self.step = Fraction(specification.step)
                self.step = specification.step
                self.num = specification.num
                self.generator = None                    # ??

                def generate_current_value():   # return next value in range
                    # Save global precision for later restoration
                    _global_precision = getcontext().prec
                    # Set SampleSpec precision
                    getcontext().prec = specification._precision
                    return_value = float(Decimal(self.start) + Decimal(self.step) * Decimal(self.current_step))
                    # Restore global precision
                    getcontext().prec = _global_precision
                    return return_value

            elif callable(specification.function):
                self.start = 0
                self.stop = None
                self.step = 1
                self.current_step = 0
                self.num = specification.num
                self.head = self.start
                self.generator = specification.function

                def generate_current_value():  # call function
                    return self.generator()

            else:
                assert False, 'PROGRAM ERROR: {} item of {} passed to specification arg of {} ' \
                              'is not an iterator or a function_type'.\
                              format(repr('function'), SampleSpec.__name__, self.__class__.__name__)

        else:
            assert False, 'PROGRAM ERROR: {} argument of {} must be one of the following: {}'.\
                          format(repr('specification'), self.__class__.__name__,
                                 (', ').join([i.__name__ for i in allowable_specs]))

        self.current_step = 0
        self.head = self.start
        self.generate_current_value = generate_current_value

    def __next__(self):
        """

        :return:
        Sample value for the current iteration.
        """
        if self.num is None:
            current_value = self.generate_current_value()
            if hasattr(self, 'stop'):
                if self.stop is not None:
                    if current_value <= self.stop:
                        return current_value
                    else:
                        raise StopIteration
            return current_value
        if self.current_step < self.num:
            current_value = self.generate_current_value()
            self.current_step += 1
            if hasattr(self, 'stop'):
                if self.stop is not None:
                    if current_value <= self.stop:
                        return current_value
                    else:
                        raise StopIteration
            return current_value

        else:
            raise StopIteration

    def __iter__(self):
        self.current_step = 0
        return self

    def __call__(self):
        return list(self)

    def reset(self, head=None):
        """Reset iterator to a specified item
        If called with None, resets to first item (if `generator <SampleIterators.generator>` is a list or
        deterministic function.
        """

        self.current_step = 0
        self.head = head or self.start
