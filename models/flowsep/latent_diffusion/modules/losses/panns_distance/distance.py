import sys

sys.path.append(
    "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio"
)

import torch.nn as nn
import torch
import os
import sys

from latent_diffusion.modules.losses.panns_distance.model.models import Cnn14_16k

MODEL = "Cnn14_16k"
CHECKPOINT_PATH = "Cnn14_16k_mAP=0.438.pth"
cmd_download_ckpt = (
    "wget -O "
    + CHECKPOINT_PATH
    + " https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
)


class Panns_distance(nn.Module):
    def __init__(self, device="cpu", metric="cos"):
        super(Panns_distance, self).__init__()
        self.panns = Cnn14_16k()
        if not os.path.exists(CHECKPOINT_PATH):
            print(cmd_download_ckpt)
            os.system(cmd_download_ckpt)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        self.metric = metric
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.panns.load_state_dict(checkpoint["model"])
        # Freeze PANNs parameters
        self.panns.eval()
        for p in self.panns.parameters():
            p.requires_grad = False

    def calculate(self, fm, fm_hat):
        distance = []
        for i, j in zip(fm, fm_hat):
            if self.metric == "cos":
                i = i.reshape(i.size(0), -1)
                j = j.reshape(j.size(0), -1)
                distance.append(self.cos(i, j)[..., None])
            else:
                distance.append(torch.mean(torch.abs(i - j)))
        if self.metric == "cos":
            distance = torch.cat(distance, dim=-1)
            return torch.mean(distance)
        else:
            return torch.mean(torch.tensor(distance))

    def forward(self, y, y_hat):
        # y: [batch, samples]
        # if y.size() != y_hat.size():
        #     print(str(y.size())  + " " + str(y_hat.size()))
        if y.size() != y_hat.size():
            min_length = min(y.size(-1), y_hat.size(-1))
            y = y[..., :min_length]
            y_hat = y_hat[..., :min_length]
        ret_dict = self.panns(y, None)
        ret_dict_hat = self.panns(y_hat, None)
        return ret_dict["feature_maps"], ret_dict_hat["feature_maps"]


if __name__ == "__main__":
    distance = Panns_distance(metric="mean")
    y = torch.randn((4, 110250))
    y_hat = torch.randn((4, 110080))
    f1, f2 = distance(y, y_hat)
    print(distance.calculate(f1, f2))
