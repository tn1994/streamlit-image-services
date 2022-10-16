import psutil


def get_memory_state_percent() -> float:
    """
    NOTES:
        if 90 > get_memory_state_percent():
            ...
        else:
            ...
    :return: percent
    """
    memory_state = psutil.virtual_memory()
    return memory_state.percent
