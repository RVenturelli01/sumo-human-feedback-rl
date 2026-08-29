"""The interactive selector: `api` holds the logic, `server` the transport."""
from .server import serve

__all__ = ["serve"]
