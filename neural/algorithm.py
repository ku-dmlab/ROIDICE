from enum import Enum

class ROIDICE(str, Enum):
    DEFAULT = "ROIDICE"

Algorithm = ROIDICE

def parse_string(name: str) -> Algorithm:
    if name in ROIDICE._value2member_map_:
        return ROIDICE(name)
    else:
        raise ValueError(f"{name} is not a supported algorithm.")
