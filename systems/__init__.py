import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    # Package isn't pip/uv-installed (e.g. running from a plain conda env)
    __version__ = "0.0.0+local"