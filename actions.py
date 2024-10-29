# actions.py
from enum import Enum

class Actions(Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    ACTION1 = 'A_BUTTON'  # Replace with actual action name
    ACTION2 = 'B_BUTTON'  # Replace with actual action name

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
