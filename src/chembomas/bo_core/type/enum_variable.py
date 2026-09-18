import enum
class INIT_METHOD(enum.Enum):
    RANDOM = enum.auto()
    EXTERNAL = enum.auto()
    CLUSTER = enum.auto()
    PSEUDO = enum.auto()