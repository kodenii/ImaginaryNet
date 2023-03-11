import random
from functools import partial
import torch

import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
from transformers import CLIPProcessor, FlaxCLIPModel
from jax import pmap
import os
import pickle
from xml.etree import ElementTree as ET
from CLIP_filter import CLIP_filter
import argparse
NUM_GPU = 8
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
seed = 0#random.randint(0, 2**32 - 1)
random.seed(seed)
# key_seed = jax.random.PRNGKey(seed)
#hide the stdout
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')


    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class p_generator(object):
    def __init__(self,backend):
        self.choice=backend
        self.dalle=None
        self.params=None
        self.vqgan=None
        self.vqgan_params=None
        self.processor=None
        self.sd=None
        self.replicate=None
        if backend=="dalle-mini":
            
            import jax.numpy as jnp
            from dalle_mini import DalleBart, DalleBartProcessor
            from vqgan_jax.modeling_flax_vqgan import VQModel
            from flax.jax_utils import replicate
            
            DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
            DALLE_COMMIT_ID = None

            VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
            VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
            self.replicate=replicate
            self.dalle, self.params = DalleBart.from_pretrained(
                DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
            )
            self.vqgan, self.vqgan_params = VQModel.from_pretrained(
                VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
            )

            self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

            self.params = replicate(self.params)
            self.vqgan_params = replicate(self.vqgan_params)
        if backend=="stablediffusion":
            from StableDiffusion import StableDiffusion
            self.sd=StableDiffusion()
    
    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,4, 5, 6, 7))
    def p_generate_dalle(
        self,tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        
        return self.dalle.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )
    
    @partial(pmap, axis_name="batch",static_broadcasted_argnums=(0))
    def p_decode_vqgan(self,indices, params):
        return self.vqgan.decode_code(indices, params=params)

    def gen_group(self,prompt, n_predictions):
        images = []
        if self.choice=="dalle-mini":
            from flax.training.common_utils import shard_prng_key
            tokenized_prompt = self.processor([prompt])
            tokenized_prompt = self.replicate(tokenized_prompt)
            import jax
            for i in range(max(n_predictions // jax.device_count(), 1)):
                # get a new key
                key, subkey = jax.random.split(jax.random.PRNGKey(random.randint(0, 2**32 - 1)))
                # generate images
                encoded_images = self.p_generate_dalle(
                    tokenized_prompt,
                    shard_prng_key(subkey),
                    self.params,
                    None,
                    None,
                    None,
                    10.0,
                )
                # remove BOS
                encoded_images = encoded_images.sequences[..., 1:]
                # decode images
                decoded_images = self.p_decode_vqgan(encoded_images, self.vqgan_params)
                decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
                for img in decoded_images:
                    images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
        if self.choice=="stablediffusion":
            for i in range(max(n_predictions//torch.cuda.device_count(), 1)):
                
                # generate images
                with HiddenPrints():
                    encoded_images = self.sd.generate(prompt,random.randint(0, 2**32 - 1))
                # decode images
                decoded_images = self.sd.decode(encoded_images)
                for img in decoded_images:
                    images.append(Image.fromarray(np.asarray(img, dtype=np.uint8)))
        imgs = [np.array(img) for img in images]
        detect_results = None#detector(imgs, size=256)

        return images, detect_results, None

def Element(name, text=None):
    tmp = ET.Element(name)
    if text is not None:
        tmp.text = str(text)
    return tmp

def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def get_xml(idx, objs,OUT_DIR):
    raw_name = f"{idx:0>6d}"

    root = Element('annotation')
    tree = ET.ElementTree(root)
    
    folder = Element('folder', "PseudoWSODv1")
    root.append(folder)

    filename = Element('filename', f'{raw_name}.jpg')
    root.append(filename)

    source = Element('source')
    database = Element('database', 'The PseudoWSODv1 Dataset')
    source.append(database)
    annotation = Element('annotation', 'PseudoWSODv1')
    source.append(annotation)
    image = Element('image', 'dalle')
    source.append(image)
    flickrid = Element('flickrid', 100000000)
    source.append(flickrid)
    root.append(source)

    owner = Element('owner')
    flickrid = Element('flickrid', 'dalle')
    owner.append(flickrid)
    name = Element('name', 'dalle')
    owner.append(name)
    root.append(owner)

    size = Element('size')
    width = Element('width', 256)
    size.append(width)
    height = Element('height', 256)
    size.append(height)
    depth = Element('depth', 3)
    size.append(depth)
    root.append(size)

    segmented = Element('segmented', 0)
    root.append(segmented)

    for obj in objs:
        object = Element('object')
        name = Element('name', obj)
        object.append(name)
        pose = Element('pose', 'Right')
        object.append(pose)
        truncated = Element('truncated', 0)
        object.append(truncated)
        difficult = Element('difficult', 0)
        object.append(difficult)

        bndbox = Element('bndbox')
        xmin = Element('xmin', 0)
        bndbox.append(xmin)
        ymin = Element('ymin', 0)
        bndbox.append(ymin)
        xmax = Element('xmax', 256)
        bndbox.append(xmax)
        ymax = Element('ymax', 256)
        bndbox.append(ymax)
        object.append(bndbox)

        root.append(object)

    __indent(root)
    tree.write(f'{OUT_DIR}/annotation/{raw_name}.xml', 'UTF-8')
    
