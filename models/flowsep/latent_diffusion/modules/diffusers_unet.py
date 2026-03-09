import sys
sys.path.append("/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/src")

from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import UNet2DModel

import torch
import torch.nn as nn

class DiffusersUNet(nn.Module):
    def __init__(self,     
        # Model itself
        in_channels=4,
        out_channels=4,
        attention_head_dim=8,
        block_out_channels=320,
        # cross attention condition
        cross_attention_dim=None, # 768
        encoder_hid_dim=None, # 1024
        # film condition
        global_additional_cond_dim=None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.block_out_channels = (block_out_channels, block_out_channels*2, block_out_channels*4, block_out_channels*8) 
        self.attention_head_dim = attention_head_dim
        self.global_additional_cond_dim = global_additional_cond_dim
        self.encoder_hid_dim = encoder_hid_dim
        if(self.cross_attention_dim is not None):
            self.down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
            # self.down_block_types=('SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D', 'DownBlock2D')
            self.mid_block_type='UNetMidBlock2DCrossAttn'
            self.up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
            # self.up_block_types=('UpBlock2D', 'SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D')
        else:
            self.down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D')
            self.mid_block_type='UNetMidBlock2DCrossAttn'
            self.up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D')
                     
        self.model = UNet2DConditionModel(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        block_out_channels=self.block_out_channels,
                        cross_attention_dim=self.cross_attention_dim,
                        attention_head_dim = self.attention_head_dim,
                        global_additional_cond_dim = self.global_additional_cond_dim,
                        encoder_hid_dim = self.encoder_hid_dim,
                        down_block_types=self.down_block_types,
                        mid_block_type=self.mid_block_type,
                        up_block_types=self.up_block_types,
                        **kwargs)
                        
        print(self.model)

    def forward(self, x, timesteps, context=None, context_attn_mask=None, y=None, **kwargs):
        if(self.cross_attention_dim is None):
            assert context is None, "The cross attention dimension is None. So you are now allowed to use context as condition"
        else:
            assert context is not None and context_attn_mask is not None, "You need to provide context matrix"

        if(self.global_additional_cond_dim is None):
            assert y is None, "The global additional cond dimension is None. So you are now allowed to use y as condition"
        else:
            assert y is not None, "You need to provide a global additional cond"
        
        if(y is not None and len(y.size()) == 3):
            y = y.squeeze(1)

        return self.model(
            sample=x,
            timestep=timesteps,
            global_condition=y,
            encoder_hidden_states = context,
            encoder_attention_mask=context_attn_mask,
        ).sample

def test():
    ###################################################
    # Have both global cond and no encoder hidden state
    unet = DiffusersUNet().cuda()

    sample_input=torch.randn(3, 4, 256, 16).cuda()
    timestep=torch.tensor([1,2,3]).cuda()
    global_input=torch.randn(3, 512).cuda()
    encoder_hidden_states = torch.randn((3, 17, 1024)).cuda()
    attention_mask = torch.zeros((3, 17)).cuda()

    output = unet(x=sample_input, timesteps=timestep, y=global_input, context=encoder_hidden_states, context_attn_mask=attention_mask)
    print(output.size())

    ###################################################
    # No global cond and no encoder hidden state
    unet = DiffusersUNet(cross_attention_dim=None, global_additional_cond_dim=None).cuda()

    sample_input=torch.randn(3, 4, 256, 16).cuda()
    timestep=torch.tensor([1,2,3]).cuda()

    output = unet(x=sample_input, timesteps=timestep)
    print(output.size())

    ###################################################
    # No encoder_hidden_state
    unet = DiffusersUNet(cross_attention_dim=None).cuda()

    sample_input=torch.randn(3, 4, 256, 16).cuda()
    timestep=torch.tensor([1,2,3]).cuda()
    global_input=torch.randn(3, 512).cuda()

    output = unet(x=sample_input, timesteps=timestep, y=global_input)
    print(output.size())

    ###################################################
    # No global cond
    unet = DiffusersUNet(global_additional_cond_dim=None).cuda()

    sample_input=torch.randn(3, 4, 256, 16).cuda()
    timestep=torch.tensor([1,2,3]).cuda()
    encoder_hidden_states = torch.randn((3, 17, 1024)).cuda()
    attention_mask = torch.zeros((3, 17)).cuda()

    output = unet(x=sample_input, timesteps=timestep, context=encoder_hidden_states, context_attn_mask=attention_mask)
    print(output.size())

    

if __name__ == "__main__":
    test()