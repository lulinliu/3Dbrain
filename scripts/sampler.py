from __future__ import annotations

import torch
import torch.nn as nn
from monai.utils import optional_import
from torch.cuda.amp import autocast
import os
import nibabel as nib
import numpy as np

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class Sampler:
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling_fn(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        conditioning: torch.Tensor,
        save_dir: str,
        file_name: str,
        output_ext: str
    ) -> torch.Tensor:
        if has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        image = input_noise
        cond_concat = conditioning.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(input_noise.shape[2:]))
        for t in progress_bar:
            with torch.no_grad():
                model_output = diffusion_model(
                    torch.cat((image, cond_concat), dim=1),
                    timesteps=torch.Tensor((t,)).to(input_noise.device).long(),
                    context=conditioning,
                )
                image, _ = scheduler.step(model_output, t, image)
        '''
        # Decode on GPU -> OOM on 5090@cabbageland
        with torch.no_grad():
            with autocast():
                sample = autoencoder_model.decode_stage_2_outputs(image)

        '''     
        with torch.inference_mode():
                    autoencoder_model_cpu = autoencoder_model.to("cpu")
                    image_cpu = image.detach().to("cpu")
                    sample = autoencoder_model_cpu.decode_stage_2_outputs(image_cpu)
        
        
        vol = sample.detach().cpu()
        if vol.ndim == 5:   # [B,C,D,H,W]
            vol = vol[0,0]
        elif vol.ndim == 4: # [C,D,H,W]
            vol = vol[0]

        vol = vol.permute(1, 2, 0).contiguous()  # [H,W,D]
        nii = nib.Nifti1Image(vol.numpy(), np.eye(4))
        os.makedirs(save_dir, exist_ok=True)
        nib.save(nii, f"{save_dir}/{file_name}.{output_ext}")   
        

        return sample
