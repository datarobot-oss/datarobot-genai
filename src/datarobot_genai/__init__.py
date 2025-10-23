from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .utils.urls import get_api_base

__all__ = ["get_api_base", "__version__"]

try:
    __version__ = version("datarobot-genai")
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.0.0"
