class ITTKException(Exception):
    """
    Vanilla base exception, for readability only. More specific ITTK-related exceptions should subclass this.
    """
    pass


class AsymmetricMatrixException(ITTKException):
    pass

