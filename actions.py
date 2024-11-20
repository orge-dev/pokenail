from enum import Enum


class Actions(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    A = "A"
    B = "B"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
