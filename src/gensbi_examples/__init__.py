from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gensbi_examples")
except PackageNotFoundError:  # pragma: no cover
    # Fallback if the package is being run without being installed
    __version__ = "unknown"  # pragma: no cover
