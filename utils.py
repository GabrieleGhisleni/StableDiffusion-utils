from huggingface_hub.community import dataclass
from IPython.utils.io import capture_output
from torch import Generator
from tqdm.auto import tqdm
import datetime as dt
from typing import *
import dataclasses
import numpy as np
import diffusers
import torch
import PIL


@dataclass
class StableDiffusionHelper:
  path: str = "./images"

  def __init__(self, pipe):
    self.pipe = pipe
    self.memory = {"prompts": []}
  

  def __generate_images_from_prompt(
      self,
      prompts: Iterable[List[str]], 
      height: Optional[int] = 512, 
      width:Optional[int] = 512,
      num_inference_steps: Optional[int] = 50,
      guidance_scale: Optional[float] = .7,
      seed: Optional[int] = None,
      save_images: bool = True
    ) -> List[PIL.Image.Image]: 

    self.memory['prompts'].extend(list(prompts))
    self.memory.update({
      "seed_generator": seed,
      "num_inference_steps": num_inference_steps,
      "guidance_scale": guidance_scale
    })

    images = []
    progress_bar = tqdm(
        self.generate_batch(prompts, 3), 
        total=len(prompts)//3, 
        desc="Generating", 
        leave=False
      )

    for prompts_batch in progress_bar:
      progress_bar.set_postfix_str(f"processing: {prompts_batch[0]}")

      with torch.autocast("cuda"):
          with capture_output() as io:
            out = self.pipe(
                prompts_batch, 
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=Generator("cuda").manual_seed(seed) if seed else None
              )
            
            images.extend(out.images)
            self.total_bar.update(len(prompts_batch))

            if save_images:
              self.save_images_individually(out.images, prompts_batch)

    return images


  def display_images_grid(
      self, imgs: List[PIL.Image.Image], 
    ) -> PIL.Image.Image:

      if len(imgs) == 1: 
        (rows, cols) = (1,1)
      elif len(imgs) % 3 != 0:
        (rows, cols) = ((len(imgs) // 3) + 1, 3)
      else: 
        (rows, cols) = ((len(imgs) // 3), 3)

      w, h = imgs[0].size
      grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
      grid_w, grid_h = grid.size
    
      for i, img in enumerate(imgs):
          grid.paste(img, box=(i%cols*w, i//cols*h))
      
      return grid



  def save_images_individually(
      self, imgs: List[PIL.Image.Image], prompts: List[str] = None
    ) -> None:

    for idx, (image, name) in enumerate(zip(imgs, prompts)):
      image.save(f"{self.generate_name(name)}_{idx}.jpg")


  def generate_batch(self, iterable, n=3):
      l = len(iterable)
      for ndx in range(0, l, n):
          yield iterable[ndx:min(ndx + n, l)]


  def generate_name(self, name):
    name = name.replace(' ', '_').replace('.', '_').replace(',', '_')
    name = name + f"_gs_{self.memory['guidance_scale']}"
    name = name + f"_nis__{self.memory['num_inference_steps']}"
    name = name + f"_sg_{self.memory['seed_generator']}"
    name = f"{int(dt.datetime.timestamp(dt.datetime.now()))}_{name}"
    name = f"{self.path}/{name}"
    return name


  def generate_images(self, prompts, GUIDANCE_SCALES, N_STEPS, w=512, h=512):
    self.total_bar = tqdm(
        total=len(prompts)*len(GUIDANCE_SCALES), desc="total"
      )
    res = {}

    for guidance_scale in GUIDANCE_SCALES:
      res[guidance_scale] = self.__generate_images_from_prompt(
          prompts, w, h, N_STEPS, guidance_scale
        )
    
    unflatten_images = list(itertools.chain.from_iterable(res.values()))
    return unflatten_images
    

  def generate_single_image(self, prompts, gs, N_STEPS, w=512, h=512):
    self.total_bar = tqdm(total=len(prompts)*len(gs), desc="total")
    out =  self.__generate_images_from_prompt(
          prompts, w, h, N_STEPS, gs
        )
    return self.display_images_grid(out)

STH = StableDiffusionHelper(PIPE)