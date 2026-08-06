import torch.nn.functional as F
from monai.transforms import MapTransform
import pathlib

class PadFromCSVd(MapTransform):
    """
    Pads image/mask to target size using precomputed padding from CSV
    """
    def __init__(self, keys, target_size=1024, pad_value=0.0):
        super().__init__(keys)
        self.target_size = target_size
        self.pad_value = pad_value
        self.IGNORE_INDEX = 255

    def __call__(self, data):
       
        d = dict(data)

        pad_top, pad_bottom = int(d["top"]), int(d["bottom"])
        pad_left, pad_right = int(d["left"]), int(d["right"])

        for k in self.keys:
            if k == "image":
                d[k] = F.pad(d[k], pad=(pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=self.pad_value)
            else:  # mask / label
                d[k] = F.pad(d[k], pad=(pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=self.IGNORE_INDEX)

        return d
