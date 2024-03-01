# -*- coding: utf-8 -*-
from enum import Enum


class ModelType(Enum):
    UNET_2D = "unet_2d"
    VNET_2D = "vnet_2d"
    R2_UNET_2D = "r2_unet_2d"
    ATT_UNET_2D = "att_unet_2d"
    TRANSUNET_2D = "transunet_2d"
    SWIN_UNET_2D = "swin_unet_2d"

    @classmethod
    def from_str(cls, model_str):
        model_str = model_str.lower()
        for model in cls:
            if model_str == model.value:
                return model
        raise ValueError("Invalid model name")


def main():
    pass


if __name__ == "__main__":
    main()
