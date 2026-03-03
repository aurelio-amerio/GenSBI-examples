from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gensbi_examples")
except PackageNotFoundError:
    # Fallback if the package is being run without being installed
    __version__ = "unknown"
