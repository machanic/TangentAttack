from abc import abstractmethod, ABC


class AttackWithDistance(Attack):
    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> Attack:
        return Repeated(self, times)

class Criterion(ABC):
    """Abstract base class to implement new criteria."""

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __call__(self, perturbed: T, outputs: T) -> T:
        """Returns a boolean tensor indicating which perturbed inputs are adversarial.
        Args:
            perturbed: Tensor with perturbed inputs ``(batch, ...)``.
            outputs: Tensor with model outputs for the perturbed inputs ``(batch, ...)``.
        Returns:
            A boolean tensor indicating which perturbed inputs are adversarial ``(batch,)``.
        """
        ...

    def __and__(self, other: "Criterion") -> "Criterion":
        return _And(self, other)


class MinimizationAttack(AttackWithDistance):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    @abstractmethod
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Runs the attack and returns perturbed inputs.
        The size of the perturbations should be as small as possible such that
        the perturbed inputs are still adversarial. In general, this is not
        guaranteed and the caller has to verify this.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            early_stop = None
        else:
            early_stop = min(epsilons)

        # run the actual attack
        xp = self.run(model, x, criterion, early_stop=early_stop, **kwargs)

        xpcs = []
        success = []
        for epsilon in epsilons:
            if epsilon is None:
                xpc = xp
            else:
                xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc)

            xpcs.append(xpc)
            success.append(is_adv)

        success_ = ep.stack(success)
        assert success_.shape == (K, N)

        xp_ = restore_type(xp)
        xpcs_ = [restore_type(xpc) for xpc in xpcs]

        if was_iterable:
            return [xp_] * K, xpcs_, restore_type(success_)
        else:
            assert len(xpcs_) == 1
            return xp_, xpcs_[0], restore_type(success_.squeeze(axis=0))

class Misclassification(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.
    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels: Any):
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.labels.shape
        is_adv = classes != self.labels
        return restore_type(is_adv)

def get_is_adversarial(
    criterion: Criterion, model: Model) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)

    return is_adversarial


def get_criterion(criterion) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)

