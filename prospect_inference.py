# %%
"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import torch.nn.functional as F

sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpu=torch.device("cpu")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512,512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/sd/sd-v1-4.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
#sampler = DDIMSampler(model)


sampler = DDIMSampler(model)
def main(prompt = '', content_dir = None,ddim_steps = 50,strength = 0.5, ddim_eta=0.0, n_iter=1, C=4, f=8, n_rows=0, scale=10.0, \
         model = None, seed=42, prospect_words = None, negative_prospect_words = None, n_samples=1, height=512, width=512, tplt_idx=0):
    
    precision="autocast"
    outdir="outputs/img2img-samples"
    seed_everything(seed)


    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) + 10
    
    if content_dir is not None:
        content_name =  content_dir.split('/')[-1].split('.')[0]
        content_image = load_img(content_dir).to(device)
        content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
        content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

        init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            #uc = model.get_learned_conditioning(batch_size * [""]) # 이렇게 넣으면 null text에 대한 임베딩 2+1개가 반환
                            uc = model.get_learned_conditioning(prompts, prospect_words=negative_prospect_words)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c= model.get_learned_conditioning(prompts, prospect_words=prospect_words)         
                                
                        

                        # img2img
#                         z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
#                         t_enc = int(strength * ddim_steps)
#                         samples = sampler.decode(z_enc, c, t_enc, 
#                                                 unconditional_guidance_scale=scale,
#                                                  unconditional_conditioning=uc,)
#                         print(z_enc.shape, uc.shape, t_enc)

#                         txt2img    
                        shape=[4, int(height/8), int(width/8)]
                        samples, intermediates = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         tplt_idx=tplt_idx,
                                                         verbose=False,
                                                         eta=ddim_eta,
                                                        #  log_every_t=50, # to see the intermediate predicted image
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc)
                    
                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
#                 output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

                toc = time.time()
    return output

# %%
model.cpu()
#model.embedding_manager.load('./logs/example_images2023-09-01T21-26-01_bird/checkpoints/embeddings_gs-3099.pt')
#model.embedding_manager.load('./logs/bear_folder2023-09-05T16-23-14_bear/checkpoints/embeddings_gs-1599.pt')
model.embedding_manager.load('./server_logs/imp_samp_bear_layout-texture-color-original_embeddings_gs-1599.pt')
model = model.to(device)
# %% ####### 개념 2개 #######
# prospect_words =  ['*', 'A cat doll']
# index: (0) layout, (1) texture, (-1) None [0,0,-1,-1,-1,-1,-1,-1,-1,-1]
# ['an image with the frame of *', 'A bear doll with a texture of *', '']
# ['','','']
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.4, \
      seed = 5, \
      height = 512, \
      width = 512, \
      tplt_idx = [0,-1,-1,-1,-1,-1,-1,-1,-1,-1], \
      prospect_words = ['a * dog', 'a * robot', 'a * robot', 'a * robot', 'a dog'], \
      negative_prospect_words = ['', '', '', '', ''], \
      model = model, \
      scale = 7.5
      )

# %% ####### 개념 3개 #######
# ['A minimalist image with * structure', 'an illustration of village in * style','A photo with the atmosphere of *','an illustration of village']
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed = 99, \
      height = 512, \
      width = 512, \
      tplt_idx = [1,1,1,1,1,1,1,1,1,1], \
      prospect_words = ['a flat cartoon illustration with * structure', 
                        'a flat cartoon illustration in * style', 
                        'a flat cartoon illustration in * mood', 
                        'a flat cartoon illustration'], \
      negative_prospect_words = ['','','',''], \
      model = model, \
      scale = 5.0
      )

# %%
p_list =                  ['A smiling rabbit doll, side and closed view', # 10 generation ends\
                           'A smiling dog, side and closed view', # 9 \
                           'A smiling dog, side and closed view', # 8 \
                           'A smiling dog, side and closed view', # 7 \
                           'A smiling dog, side and closed view', # 6 \
                           'A smiling dog, side and closed view', # 5 \
                           'A smiling dog, side and closed view', # 4 \
                           'A smiling dog, side and closed view', # 3 \
                           'A smiling dog, side and closed view', # 2 \
                           'A smiling dog, side and closed view', # 1 generation starts\
                          ]
new_list = p_list

for i in range(len(p_list)):
    new_list[i] = new_list[i] + ' with a style of *'
    print(new_list)
    main(prompt = '*',
         ddim_steps = 50,
         strength = 0.6,
         seed=171,
         height = 512,
         width = 512,
         prospect_words = new_list,
         model = model
         ).show()
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=111, \
      height = 512, \
      width = 512, \
      prospect_words =    ['A smiling dog, side and closed view', # 10 generation ends\
                           'A smiling dog, side and closed view', # 9 \
                           'A smiling dog, side and closed view', # 8 \
                           'A smiling dog, side and closed view', # 7 \
                           'A smiling dog, side and closed view', # 6 \
                           'A smiling dog, side and closed view', # 5 \
                           'A smiling dog, side and closed view', # 4 \
                           'A smiling dog, side and closed view', # 3 \
                           'A smiling dog, side and closed view', # 2 \
                           'A smiling dog, side and closed view', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=171, \
      height = 512, \
      width = 512, \
      prospect_words =    ['A smiling dog, side and closed view with a style of *', # 10 generation ends\
                           'A smiling dog, side and closed view with a style of *', # 9 \
                           'A smiling dog, side and closed view', # 8 \
                           'A smiling dog, side and closed view', # 7 \
                           'A smiling dog, side and closed view', # 6 \
                           'A smiling dog, side and closed view', # 5 \
                           'A smiling dog, side and closed view', # 4 \
                           'A smiling dog, side and closed view', # 3 \
                           'A smiling dog, side and closed view', # 2 \
                           'A smiling dog, side and closed view', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=111, \
      height = 512, \
      width = 512, \
      prospect_words =    ['A * flamingo', # 10 generation ends\
                           'A * flamingo', # 9 \
                           'A * flamingo', # 8 \
                           'A * flamingo', # 7 \
                           'A * flamingo', # 6 \
                           'A * flamingo', # 5 \
                           'A * flamingo', # 4 \
                           'A flamingo', # 3 \
                           'A flamingo', # 2 \
                           'A flamingo', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=42, \
      height = 512, \
      width = 768, \
      prospect_words = ['a teddy * walking in times square', # 10 generation ends\
                           'a teddy * walking in times square', # 9 \
                           'a teddy * walking in times square', # 8 \
                           'a teddy * walking in times square', # 7 \
                           'a teddy * walking in times square', # 6 \
                           'a teddy * walking in times square', # 5 \
                           'a teddy * walking in times square', # 4 \
                           'a teddy * walking in times square', # 3 \
                           'a teddy walking in times square', # 2 \
                           'a teddy walking in times square', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%
main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=8888, \
      height = 512, \
      width = 512, \
      prospect_words = ['a * bird singing on the yellow tree, closed up', # 10 generation ends\
                           'a * bird singing on the yellow tree, closed up', # 9 \
                           'a * bird singing on the yellow tree, closed up', # 8 \
                           'a * bird singing on the yellow tree, closed up', # 7 \
                           'a * bird singing on the yellow tree, closed up', # 6 \
                           'a * bird singing on the yellow tree, closed up', # 5 \
                           'a * bird singing on the yellow tree, closed up', # 4 \
                           'a * bird singing on the yellow tree, closed up', # 3 \
                           'a * bird singing on the yellow tree, closed up', # 2 \
                           'a bird singing on the yellow tree, closed up', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%
def custom_main(prompt = '', content_dir = None,ddim_steps = 50,strength = 0.5, ddim_eta=0.0, n_iter=1, C=4, f=8, n_rows=0, scale=10.0, \
         model = None, seed=42, prospect_words = None, n_samples=1, height=512, width=512, tplt_idx=0):
    
    precision="autocast"
    seed_everything(seed)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c= model.get_learned_conditioning(prompts, prospect_words=prospect_words)         
            
                        

                        # img2img
#                         z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
#                         t_enc = int(strength * ddim_steps)
#                         samples = sampler.decode(z_enc, c, t_enc, 
#                                                 unconditional_guidance_scale=scale,
#                                                  unconditional_conditioning=uc,)
#                         print(z_enc.shape, uc.shape, t_enc)

#                         txt2img    
                        shape=[4, int(height/8), int(width/8)]
                        samples, intermediates = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         tplt_idx=tplt_idx,
                                                         verbose=False,
                                                         eta=ddim_eta,
                                                        #  log_every_t=50, # to see the intermediate predicted image
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc)
                    
                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
#                 output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

                toc = time.time()
    return output
# %%
custom_main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=53, \
      height = 512, \
      width = 512, \
      prospect_words = ['*', # 10 generation ends\
                           '*', # 9 \
                           '*', # 8 \
                           '*', # 7 \
                           '*', # 6 \
                           '*', # 5 \
                           '*', # 4 \
                           '*', # 3 \
                           '*', # 2 \
                           '*', # 1 generation starts\
                          ], \
      model = model,\
      )

# %%

main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=359, \
      height = 512, \
      width = 512, \
      prospect_words =    ['A * squirrel in a forest, side and very closed view', # 10 generation ends\
                           'A squirrel in a forest, side and very closed view', # 9 \
                           'A squirrel in a forest, side and very closed view', # 8 \
                           'A squirrel in a forest, side and very closed view', # 7 \
                           'A squirrel in a forest, side and very closed view', # 6 \
                           'A squirrel in a forest, side and very closed view', # 5 \
                           'A squirrel in a forest, side and very closed view', # 4 \
                           'A squirrel in a forest, side and very closed view', # 3 \
                           'A * squirrel in a forest, side and very closed view', # 2 \
                           'A squirrel in a forest, side and very closed view', # 1 generation starts\
                          ], \
      model = model,\
      )
# %%

main(prompt = '*', \
      ddim_steps = 50, \
      strength = 0.6, \
      seed=359, \
      height = 512, \
      width = 512, \
      prospect_words =    ['A * fluffy bear doll, side and very closed view', # 10 generation ends\
                           'A * fluffy bear doll, side and very closed view', # 9 \
                           'A * fluffy bear doll, side and very closed view', # 8 \
                           'A * fluffy bear doll, side and very closed view', # 7 \
                           'A * fluffy bear doll, side and very closed view', # 6 \
                           'A * fluffy bear doll, side and very closed view', # 5 \
                           'A * fluffy bear doll, side and very closed view', # 4 \
                           'A * fluffy bear doll, side and very closed view', # 3 \
                           'A * fluffy bear doll, side and very closed view', # 2 \
                           'A fluffy bear doll, side and very closed view', # 1 generation starts\
                          ], \
      model = model,\
      )