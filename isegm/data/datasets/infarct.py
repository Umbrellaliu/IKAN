import cv2
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import os
import matplotlib.pyplot as plt


class InfarctAISDDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(InfarctAISDDataset, self).__init__(**kwargs)
        assert split in {'train', 'test'}
        self.name = 'InfarctAISD'
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / f'infarct_aisd_{split}'


        self.dataset_samples = list(self._images_path.glob('*.png'))


    def get_sample(self, index):
        image_path = self.dataset_samples[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = image[:, :, 1]  # Extract the G channel as the label
        image = image[:, :, 0]  # Extract the R channel as the image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, (256, 256))
        gt = cv2.resize(gt, (256, 256))
        gt = (gt > 128).astype(np.int32)

        return DSample(image, gt, objects_ids=[1], sample_id=index)




