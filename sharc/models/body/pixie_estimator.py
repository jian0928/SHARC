import torch


class PIXIEEstimator:
    def __init__(self, cfg=None, device=None):
        self.cfg = cfg
        self.device = device
        self.model = None

    def load(self):
        from pixie import PIXIE
        self.model = PIXIE(self.cfg)

    @torch.no_grad()
    def estimate(self, image, bbox=None):
        if self.model is None:
            self.load()
        return self.model(image, bbox)

    def extract_params(self, param_dict):
        return {
            'betas': param_dict.get('betas'),
            'body_pose': param_dict.get('body_pose'),
            'global_orient': param_dict.get('global_orient'),
            'transl': param_dict.get('transl'),
            'jaw_pose': param_dict.get('jaw_pose'),
            'leye_pose': param_dict.get('leye_pose'),
            'reye_pose': param_dict.get('reye_pose'),
            'left_hand_pose': param_dict.get('left_hand_pose'),
            'right_hand_pose': param_dict.get('right_hand_pose'),
        }
