from termcolor import colored
from enum import Enum
from datetime import datetime
import os

class LogLevel(Enum):
    Trace = 0
    Debug = 1
    Info = 2
    Warning = 3
    Error = 4
    Disabled = 5

class Logger:
    """
    A logging class that supports different levels of logging and colored output.
    Levels: Trace, Debug, Info, Warning, Error, Disabled.
    """
    def __init__(self, min_log_level=LogLevel.Info):
        self.min_log_level = min_log_level

    def log_message(self, level, message):
        """
        Logs a message at the specified level if it meets the minimum log level requirement.
        :param level: LogLevel, the level of the log message.
        :param message: str, the message to log.
        """
        if level.value < self.min_log_level.value:
            return

        color_map = {
            LogLevel.Error: 'red',
            LogLevel.Warning: 'yellow',
            LogLevel.Info: 'blue',
            LogLevel.Debug: 'cyan',
            LogLevel.Trace: 'magenta'
        }
        print(colored(message, color_map.get(level, 'white')))

    def error(self, message):
        self.log_message(LogLevel.Error, message)

    def warning(self, message):
        self.log_message(LogLevel.Warning, message)

    def info(self, message):
        self.log_message(LogLevel.Info, message)

    def debug(self, message):
        self.log_message(LogLevel.Debug, message)

    def trace(self, message):
        self.log_message(LogLevel.Trace, message)

def make_dir(dir_list):
    """
    Creates a directory from a list of directory segments if it does not already exist.
    :param dir_list: list of strings, segments of the directory path.
    :return: str, the full path of the created directory.
    """
    dir_path = os.path.join(*dir_list)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def print_recursive_structure_and_content(obj, indent=0, max_elements=10):
    """
    Recursively prints the structure and content of the given object, detailing its type and elements.
    Handles dictionaries, lists, sets, and tuples. Limits the number of elements printed for collections.
    :param obj: Object to be printed.
    :param indent: Indentation level for pretty printing, increased recursively.
    :param max_elements: Maximum number of elements to print for each collection.
    """
    if isinstance(obj, dict):
        print(" " * indent + f"Dictionary with {len(obj)} elements:")
        for key, value in obj.items():
            print(" " * indent + f"Key: {key}, Value:")
            print_recursive_structure_and_content(value, indent + 4, max_elements)
    elif isinstance(obj, (list, set, tuple)):
        collection_type = type(obj).__name__
        print(" " * indent + f"{collection_type} with {len(obj)} elements:")
        for i, item in enumerate(obj):
            if max_elements is not None and i >= max_elements:
                print(" " * indent + f"... {len(obj) - i} more elements")
                break
            print_recursive_structure_and_content(item, indent + 4, max_elements)
    else:
        print(" " * indent + f"{type(obj)}: {obj}")

def find_common_and_uncommon_elements(dict_list, key):
    """
    Identifies common and uncommon elements across all dictionaries in the provided dict_list.
    Each dictionary in dict_list is expected to have a specified key with a list of elements.

    :param dict_list: List of dictionaries, each expected to contain the specified key.
    :param key: The key to look for in each dictionary to extract the elements for analysis.
    :return: A tuple containing two sets: (common_elements, uncommon_elements).
    """
    if not dict_list:
        return set(), set()  # Return empty sets if dict_list is empty

    common_elements = set(dict_list[0].get(key, []))
    uncommon_elements = set()

    for entry in dict_list[1:]:
        entry_elements = set(entry.get(key, []))
        common_elements.intersection_update(entry_elements)
        uncommon_elements.update(entry_elements)

    uncommon_elements.difference_update(common_elements)

    return common_elements, uncommon_elements
