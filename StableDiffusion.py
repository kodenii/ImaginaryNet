from scripts.txt2img import *
from omegaconf import OmegaConf
import os
import wget
class StableDiffusion(object):
    def __init__(self):
        config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
        if not os.path.isfile("model.ckpt"):
            print("downloading the weights..., please wait for a moment...")
            wget.download("https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt","model.ckpt")
            print('\n')
        self.model = load_model_from_config(config, "model.ckpt")
    
    def generate(self,prompt, key, params=None, top_k=None, top_p=None, temperature=None, condition_scale=9.0):

        seed_everything(key)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = self.model.to(device)
        sampler = PLMSSampler(model)
        # sampler = DPMSolverSampler(model)
        # sampler = DDIMSampler(model)
        batch_size = 1
        n_rows = batch_size
        data = [[prompt]]
        start_code = None

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in range(1):
                        for prompts in data:
                            uc = None
                            if condition_scale != 1.0:
                                uc = model.get_learned_conditioning([""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [4, 256 // 4, 256 // 4]
                            samples_ddim, _ = sampler.sample(S=50,
                                                            conditioning=c,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=condition_scale,
                                                            unconditional_conditioning=uc,
                                                            eta=0.0,
                                                            x_T=start_code)
        return samples_ddim
                            
                            

    def decode(self,indices):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = self.model.to(device)
        x_samples_ddim = model.decode_first_stage(indices)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_samples_ddim = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_samples_ddim=[255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c') for x_sample in x_samples_ddim]
        return x_samples_ddim