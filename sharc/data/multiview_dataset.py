import torch
from torch.utils.data import Dataset


class MultiviewDataset(Dataset):
    def __init__(self, base_dataset=None, multiview_angles=None, face_crop=None, transform=None):
        self.base_dataset = base_dataset
        self.multiview_angles = multiview_angles
        self.face_crop = face_crop
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset) * len(self.multiview_angles)

    def __getitem__(self, index):
        sample_idx = index // len(self.multiview_angles)
        view_idx = index % len(self.multiview_angles)
        sample = self.base_dataset[sample_idx]
        image = sample['image']
        target_normal = sample['target_normals'][view_idx]
        target_color = sample['target_colors'][view_idx]
        smplx_normal = sample['smplx_normals'][view_idx]
        smplx_silhouette = sample['smplx_silhouettes'][view_idx]
        if self.transform is not None:
            image = self.transform(image)
            target_normal = self.transform(target_normal)
            target_color = self.transform(target_color)
            smplx_normal = self.transform(smplx_normal)
            smplx_silhouette = self.transform(smplx_silhouette)
        face_crop = None
        face_mask = None
        if self.face_crop is not None:
            bbox = sample.get('face_bbox')
            face_crop, bbox = self.face_crop.crop(target_color, bbox)
            face_mask = self.face_crop.make_mask(target_color.shape[-2:], bbox)
        return {
            'image': image,
            'smplx_normal': smplx_normal,
            'smplx_silhouette': smplx_silhouette,
            'target_normal': target_normal,
            'target_color': target_color,
            'face_crop': face_crop,
            'face_mask': face_mask,
            'view_index': view_idx,
        }
