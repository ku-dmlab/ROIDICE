from enum import Enum

class Hopper(str, Enum):
    MEDIUM = "hopper-medium-v2"
    MEDIUM_REPLAY = "hopper-medium-replay-v2"
    MEDIUM_EXPERT = "hopper-medium-expert-v2"
    EXPERT = "hopper-expert-v2"


class Halfcheetah(str, Enum):
    MEDIUM = "halfcheetah-medium-v2"
    MEDIUM_REPLAY = "halfcheetah-medium-replay-v2"
    MEDIUM_EXPERT = "halfcheetah-medium-expert-v2"
    EXPERT = "halfcheetah-expert-v2"


class Walker2D(str, Enum):
    MEDIUM = "walker2d-medium-v2"
    MEDIUM_REPLAY = "walker2d-medium-replay-v2"
    MEDIUM_EXPERT = "walker2d-medium-expert-v2"
    EXPERT = "walker2d-expert-v2"


class Finance(str, Enum):
    MEDIUM = "finance-medium-100"
    HIGH = "finance-high-100"


MujocoEnvironmentName = Hopper | Halfcheetah | Walker2D
FinanceEnvironmentName = Finance
EnvironmentName = MujocoEnvironmentName | FinanceEnvironmentName


def parse_string(name: str) -> EnvironmentName:
    match name.split("-"):
        case ["hopper", *_]:
            return Hopper(name)
        case ["halfcheetah", *_]:
            return Halfcheetah(name)
        case ["walker2d", *_]:
            return Walker2D(name)
        case ["finance", *_]:
            return Finance(name)
        case _:
            raise ValueError(f"{name} is not supported.")
    
