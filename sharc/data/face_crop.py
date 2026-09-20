import torch
import torch.nn.functional as F


class FaceCrop:
    def __init__(self, detector=None, crop_size=None, margin=None):
        self.detector = detector
        self.crop_size = crop_size
        self.margin = margin

    def detect_bbox(self, image):
        if self.detector is None:
            raise NotImplementedError
        return self.detector(image)

    def crop(self, image, bbox=None):
        if bbox is None:
            bbox = self.detect_bbox(image)
        x1, y1, x2, y2 = bbox
        if self.margin is not None:
            w = x2 - x1
            h = y2 - y1
            x1 = x1 - self.margin * w
            y1 = y1 - self.margin * h
            x2 = x2 + self.margin * w
            y2 = y2 + self.margin * h
        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = int(min(x2, image.shape[-1]))
        y2 = int(min(y2, image.shape[-2]))
        crop = image[..., y1:y2, x1:x2]
        if self.crop_size is not None:
            crop = F.interpolate(crop.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False).squeeze(0)
        return crop, (x1, y1, x2, y2)

    def make_mask(self, image_size, bbox=None):
        h, w = image_size
        mask = torch.zeros((1, 1, h, w))
        if bbox is None:
            return mask
        x1, y1, x2, y2 = bbox
        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = int(min(x2, w))
        y2 = int(min(y2, h))
        mask[..., y1:y2, x1:x2] = 1.0
        return mask
