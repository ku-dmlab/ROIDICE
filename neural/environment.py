from enum import Enum


class AntMaze(str, Enum):
    UMAZE = "antmaze-umaze-v0"
    UMAZE_DIVERSE = "antmaze-umaze-diverse-v0"
    MEDIUM_DIVERSE = "antmaze-medium-diverse-v0"
    MEDIUM_PLAY = "antmaze-medium-play-v0"
    LARGE_DIVERE = "antmaze-large-diverse-v0"
    LARGE_PLAY = "antmaze-large-play-v0"

class PointMaze(str, Enum):
    UMAZE = "maze2d-umaze-v1"

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


class Point(str, Enum):
    GOAL = "Safexp-PointGoal1-v0"
    BUTTON = "Safexp-PointButton1-v0"
    PUSH = "Safexp-PointPush1-v0"


class Car(str, Enum):
    GOAL = "Safexp-CarGoal1-v0"
    BUTTON = "Safexp-CarButton1-v0"
    PUSH = "Safexp-CarPush1-v0"


MujocoEnvironmentName = Hopper | Halfcheetah | Walker2D
SafetyGymEnvironmentName = Point | Car
MazeEnvironmentName = AntMaze | PointMaze
EnvironmentName = MazeEnvironmentName | MujocoEnvironmentName | SafetyGymEnvironmentName


def parse_string(name: str) -> EnvironmentName:
    match name.split("-"):
        case ["maze2d", *_]:
            return PointMaze(name)
        case ["hopper", *_]:
            return Hopper(name)
        case ["halfcheetah", *_]:
            return Halfcheetah(name)
        case ["walker2d", *_]:
            return Walker2D(name)
        case ["Safexp", env_name, _] if env_name.startswith("Point"):
            return Point(name)
        case ["Safexp", env_name, _] if env_name.startswith("Car"):
            return Car(name)
        case _:
            raise ValueError(f"{name} is not supported.")
