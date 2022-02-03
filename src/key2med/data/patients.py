from dataclasses import dataclass
from typing import List

from key2med.data import ImagePath


@dataclass
class Study:
    name: str
    images: List[ImagePath]

    @property
    def number(self):
        return int(self.name.split("study")[-1])


@dataclass
class Patient:
    name: str
    studies: List[Study]
