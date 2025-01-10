from typing import List, Optional

UNITS = ["bytes", "kB", "MB", "GB", "TB", "PB", "EB"]


def get_readable_size(size: int, units: Optional[List[str]] = None) -> str:
    """Return a human readable string representation of bytes."""
    units = units or UNITS
    return (
        str(size) + " " + units[0]
        if size < 1024 or len(units) == 1
        else get_readable_size(size >> 10, units[1:])
    )
