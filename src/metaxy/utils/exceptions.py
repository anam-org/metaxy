class MetaxyError(Exception):
    """Base class for all errors thrown by the Metaxy framework.

    Users should not subclass this base class for their own exceptions.
    """

    @property
    def is_user_code_error(self):
        """Returns true if this error is attributable to user code."""
        return False


class MetaxyInvariantViolationError(MetaxyError):
    """Indicates the user has violated a well-defined invariant that can only be enforced
    at runtime.
    """


class MetaxyEmptyCodeVersionError(MetaxyInvariantViolationError):
    """Indicates that an empty code version was provided where it is not allowed.

    Code version must be a non-empty string.
    """
