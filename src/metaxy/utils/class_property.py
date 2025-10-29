class ClassPropertyDescriptor:
    """Descriptor to enable property-like access for class methods."""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        return self.fget(objtype)
