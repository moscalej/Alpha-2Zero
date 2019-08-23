"""

"""

import os
import yaml

class Looger:
    def __init__(self, path:str):
        """
        :param path: Path to the file the loger will create or use
        :type path:
        """
        self.file = yaml.full_load(open(path, 'w'))

    def update_arena(self, new_network, old_):
        pass