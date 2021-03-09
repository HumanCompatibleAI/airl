class IdentityCounter:
    """Identity callable that counts all calls.

    This is useful for making sure that the *augmentation_fn keyword arguments
    for BC and GAIL/AIRL are actually working."""

    def __init__(self):
        self.ncalls = 0

    def __call__(self, x):
        self.ncalls += 1
        return x
