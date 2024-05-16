from enum import Enum

class OptiDICE(str, Enum):
    DEFAULT = "OptiDICE"
    FULL = "FullOptiDICE"

class COptiDICE(str, Enum):
    DEFAULT = "COptiDICE"
    UB = "UBCOptiDICE" # upper bound
    LAMBDA = "LambdaCOptiDICE"

class ROIDICE(str, Enum):
    DEFAULT = "ROIDICE"

class BC(str, Enum):
    DEFAULT = "BC"


UnconstrainedRL = OptiDICE
ConstrainedRL = COptiDICE | ROIDICE
UnconstrainedIL = BC
Algorithm = UnconstrainedRL | ConstrainedRL | UnconstrainedIL


def parse_string(name: str) -> Algorithm:
    if name in OptiDICE._value2member_map_:
        return OptiDICE(name)
    elif name in COptiDICE._value2member_map_:
        return COptiDICE(name)
    elif name in ROIDICE._value2member_map_:
        return ROIDICE(name)
    elif name in BC._value2member_map_:
        return BC(name)
    else:
        raise ValueError(f"{name} is not a supported algorithm.")
