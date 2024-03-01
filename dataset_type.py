# -*- coding: utf-8 -*-
from enum import Enum


class DatasetType(Enum):
    ISBI2015 = "ISBI2015"
    MSSEG2016 = "MSSEG2016"
    ALL = "ALL"

    @classmethod
    def from_str(cls, dataset_str):
        dataset_str = dataset_str.upper()
        for dataset in cls:
            if dataset_str == dataset.name or dataset_str == dataset.value.upper():
                return dataset
        raise ValueError("Invalid dataset string")


def main():
    pass


if __name__ == "__main__":
    main()
