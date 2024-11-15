# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import mmcv
print(mmcv.__version__)
from mmengine.model.base_module import BaseModule # from mmcv.runner import BaseModule


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head."""

    def __init__(self, init_cfg=None):
        super(BaseHead, self).__init__(init_cfg)

    @abstractmethod
    def forward_train(self, x, gt_label, **kwargs):
        pass
