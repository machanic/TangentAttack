"""
Provides classes to measure the distance between inputs.

Distances
---------

.. autosummary::
   :nosignatures:

   MeanSquaredDistance
   MeanAbsoluteDistance
   Linfinity
   L0

Aliases
-------

.. autosummary::
   :nosignatures:

   MSE
   MAE
   Linf

Base class
----------

To implement a new distance, simply subclass the :class:`Distance` class and
implement the :meth:`_calculate` method.

.. autosummary::
   :nosignatures:

   Distance

"""
from __future__ import division
import sys
import abc
import torch
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})

import functools
from numbers import Number

from torch.nn import functional as F
import numpy as np



@functools.total_ordering
class Distance(ABC):
    """Base class for distances.

    This class should be subclassed when implementing
    new distances. Subclasses must implement _calculate.

    """

    def __init__(
            self,
            reference=None,
            other=None,
            bounds=None,
            value=None):

        if value is not None:
            # alternative constructor
            assert isinstance(value, Number)
            assert reference is None
            assert other is None
            assert bounds is None
            self.reference = None
            self.other = None
            self._bounds = None
            self._value = value
            self._gradient = None
        else:
            # standard constructor
            self.reference = reference
            self.other = other
            self._bounds = bounds
            self._value, self._gradient = self._calculate()

        assert self._value is not None

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @abstractmethod
    def _calculate(self):
        """Returns distance and gradient of distance w.r.t. to self.other"""
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '{} = {:.6e}'.format(self.name(), self._value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')
        return self.value == other.value

    def __lt__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')
        return self.value < other.value


class MeanSquaredDistance(Distance):
    """Calculates the mean squared error between two inputs.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        n = self.reference.numel()
        f = n * (max_ - min_)**2

        diff = self.other - self.reference
        value = torch.dot(diff.view(-1), diff.view(-1)).item() / f

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self._g_diff / (self._g_f / 2)
        return self._gradient

    def __str__(self):
        return 'normalized MSE = {:.2e}'.format(self._value)


MSE = MeanSquaredDistance


class MeanAbsoluteDistance(Distance):
    """Calculates the mean absolute error between two inputs.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = torch.mean(torch.abs(diff)).type(torch.float64)
        n = self.reference.size
        gradient = 1 / n * torch.sign(diff) / (max_ - min_)
        return value, gradient

    def __str__(self):
        return 'normalized MAE = {:.2e}'.format(self._value)


MAE = MeanAbsoluteDistance


class Linfinity(Distance):
    """Calculates the L-infinity norm of the difference between two inputs.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = torch.max(torch.abs(diff)).type(torch.float64)
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError

    def __str__(self):
        return 'normalized Linf distance = {:.2e}'.format(self._value)


Linf = Linfinity


class L0(Distance):
    """Calculates the L0 norm of the difference between two inputs.

    """

    def _calculate(self):
        diff = self.other - self.reference
        value = torch.sum(diff != 0)
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError

    def __str__(self):
        return 'L0 distance = {}'.format(self._value)




"""
Provides classes that define what is adversarial.

Criteria
--------

We provide criteria for untargeted and targeted adversarial attacks.

.. autosummary::
   :nosignatures:

   Misclassification
   TopKMisclassification
   OriginalClassProbability
   ConfidentMisclassification

.. autosummary::
   :nosignatures:

   TargetClass
   TargetClassProbability

Examples
--------

Untargeted criteria:

>>> from foolbox.criteria import Misclassification
>>> criterion1 = Misclassification()

>>> from foolbox.criteria import TopKMisclassification
>>> criterion2 = TopKMisclassification(k=5)

Targeted criteria:

>>> from foolbox.criteria import TargetClass
>>> criterion3 = TargetClass(22)

>>> from foolbox.criteria import TargetClassProbability
>>> criterion4 = TargetClassProbability(22, p=0.99)

Criteria can be combined to create a new criterion:

>>> criterion5 = criterion2 & criterion3

"""



class Criterion(ABC):
    """Base class for criteria that define what is adversarial.

    The :class:`Criterion` class represents a criterion used to
    determine if predictions for an image are adversarial given
    a reference label. It should be subclassed when implementing
    new criteria. Subclasses must implement is_adversarial.

    """

    def name(self):
        """Returns a human readable name that uniquely identifies
        the criterion with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the criterion
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial given
        a reference label.

        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.

        Returns
        -------
        bool
            True if an image with the given predictions is an adversarial
            example when the ground-truth class is given by label, False
            otherwise.

        """
        raise NotImplementedError

    def __and__(self, other):
        return CombinedCriteria(self, other)


class CombinedCriteria(Criterion):
    """Meta criterion that combines several criteria into a new one.

    Considers inputs as adversarial that are considered adversarial
    by all sub-criteria that are combined by this criterion.

    Instead of using this class directly, it is possible to combine
    criteria like this: criteria1 & criteria2

    Parameters
    ----------
    *criteria : variable length list of :class:`Criterion` instances
        List of sub-criteria that will be combined.

    Notes
    -----
    This class uses lazy evaluation of the criteria in the order they
    are passed to the constructor.

    """

    def __init__(self, *criteria):
        super(CombinedCriteria, self).__init__()
        self._criteria = criteria

    def name(self):
        """Concatenates the names of the given criteria in alphabetical order.

        If a sub-criterion is itself a combined criterion, its name is
        first split into the individual names and the names of the
        sub-sub criteria is used instead of the name of the sub-criterion.
        This is done recursively to ensure that the order and the hierarchy
        of the criteria does not influence the name.

        Returns
        -------
        str
            The alphabetically sorted names of the sub-criteria concatenated
            using double underscores between them.

        """
        names = (criterion.name() for criterion in self._criteria)
        return '__'.join(sorted(names))

    def is_adversarial(self, predictions, label):
        for criterion in self._criteria:
            if not criterion.is_adversarial(predictions, label):
                # lazy evaluation
                return False
        return True


class Misclassification(Criterion):
    """Defines adversarials as inputs for which the predicted class
    is not the original class.

    See Also
    --------
    :class:`TopKMisclassification`

    Notes
    -----
    Uses `numpy.argmax` to break ties.

    """

    def name(self):
        return 'Top1Misclassification'

    def is_adversarial(self, predictions, label):
        top1 = torch.argmax(predictions).item()
        return top1 != label


class ConfidentMisclassification(Criterion):
    """Defines adversarials as inputs for which the probability
    of any class other than the original is above a given threshold.

    Parameters
    ----------
    p : float
        The threshold probability. If the probability of any class
        other than the original is at least p, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.

    """

    def __init__(self, p):
        super(ConfidentMisclassification, self).__init__()
        assert 0 <= p <= 1
        self.p = p

    def name(self):
        return '{}-{:.04f}'.format(self.__class__.__name__, self.p)

    def is_adversarial(self, predictions, label):
        top1 = torch.argmax(predictions)
        probabilities = F.softmax(predictions)
        return (torch.max(probabilities) >= self.p) and (top1 != label)


class TopKMisclassification(Criterion):
    """Defines adversarials as inputs for which the original class is
    not one of the top k predicted classes.

    For k = 1, the :class:`Misclassification` class provides a more
    efficient implementation.

    Parameters
    ----------
    k : int
        Number of top predictions to which the reference label is
        compared to.

    See Also
    --------
    :class:`Misclassification` : Provides a more effcient implementation
        for k = 1.

    Notes
    -----
    Uses `numpy.argsort` to break ties.

    """

    def __init__(self, k):
        super(TopKMisclassification, self).__init__()
        self.k = k

    def name(self):
        return 'Top{}Misclassification'.format(self.k)

    def is_adversarial(self, predictions, label):
        topk = torch.argsort(predictions)[-self.k:]
        return label not in topk


class TargetClass(Criterion):
    """Defines adversarials as inputs for which the predicted class
    is the given target class.

    Parameters
    ----------
    target_class : int
        The target class that needs to be predicted for an image
        to be considered an adversarial.

    Notes
    -----
    Uses `numpy.argmax` to break ties.

    """

    def __init__(self, target_class=None):
        super(TargetClass, self).__init__()
        self._target_class = target_class

    def target_class(self):
        return self._target_class

    def name(self):
        return '{}-{}'.format(self.__class__.__name__, self.target_class())

    def is_adversarial(self, predictions, label=None):
        top1 = torch.argmax(predictions,dim=-1).item()
        return top1 == self.target_class()  # target class 其实是true label


class OriginalClassProbability(Criterion):
    """Defines adversarials as inputs for which the probability
    of the original class is below a given threshold.

    This criterion alone does not guarantee that the class
    predicted for the adversarial image is not the original class
    (unless p < 1 / number of classes). Therefore, it should usually
    be combined with a classifcation criterion.

    Parameters
    ----------
    p : float
        The threshold probability. If the probability of the
        original class is below this threshold, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.

    """

    def __init__(self, p):
        super(OriginalClassProbability, self).__init__()
        assert 0 <= p <= 1
        self.p = p

    def name(self):
        return '{}-{:.04f}'.format(self.__class__.__name__, self.p)

    def is_adversarial(self, predictions, label):
        probabilities = F.softmax(predictions)
        return probabilities[label] < self.p


class TargetClassProbability(Criterion):
    """Defines adversarials as inputs for which the probability
    of a given target class is above a given threshold.

    If the threshold is below 0.5, this criterion does not guarantee
    that the class predicted for the adversarial image is not the
    original class. In that case, it should usually be combined with
    a classification criterion.

    Parameters
    ----------
    target_class : int
        The target class for which the predicted probability must
        be above the threshold probability p, otherwise the image
        is not considered an adversarial.
    p : float
        The threshold probability. If the probability of the
        target class is above this threshold, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.

    """

    def __init__(self, target_class, p):
        super(TargetClassProbability, self).__init__()
        self._target_class = target_class
        assert 0 <= p <= 1
        self.p = p

    def target_class(self):
        return self._target_class

    def name(self):
        return '{}-{}-{:.04f}'.format(
            self.__class__.__name__, self.target_class(), self.p)

    def is_adversarial(self, predictions, label):
        probabilities = softmax(predictions)
        return probabilities[self.target_class()] > self.p
