import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    '{}'
]
imagenet_dual_templates_small = [
    '{} {}'
]

"""
template_layout = [
    'A bear doll shaped of {}',
    'A bear doll in the shape of {}',
    'A bear doll in a shape of {}',
    'A bear doll with the shape of {}',
    'A bear doll with a shape of {}',
    
    'A bear doll with the composition of {}',
    'A bear doll with a composition of {}',
    
    'A bear doll with the frame of {}',
    'A bear doll with a frame of {}',
    
    'A bear doll with the structure of {}',
    'A bear doll with a structure of {}',
]
"""
template_layout = [
    'an image shaped of {}',
    'an image with the shape of {}',
    'an image with a shape of {}',
    'an image with a {} shape',
    
    'an image with the composition of {}',
    'an image with a composition of {}',
    'an image with a {} composition',

    'an image with the frame of {}',
    'an image with a frame of {}',
    'an image with a {} frame',
    
    'an image with the structure of {}',
    'an image with a structure of {}',
    'an image with a {} structure'
]

template_texture = [
    'A bear doll with the texture of {}',
    'A bear doll with a texture of {}',
    'A bear doll with a {} texture',
    '{} textured bear doll',
    
    'A bear doll with the style of {}',
    'A bear doll with a style of {}',
    'A bear doll with a {} style',
    '{} style bear doll',

    'A bear doll with the material of {}',
    'A bear doll with a material of {}',
    'A bear doll with a {} material',
    '{} material bear doll',

    'A bear doll with the touch of {}',
    'A bear doll with a touch of {}',
    'A bear doll with a {} touch',
    '{} touch bear doll'
]

template_atmosphere = [
    'A photo with the atmosphere of {}',
    'A photo with an atmosphere of {}',
    'A photo with an {} atmosphere',
    '{} atmosphere photo',
    
    'A photo with the mood of {}',
    'A photo with a mood of {}',
    'A photo with a {} mood',
    '{} mood photo',
    
    'A photo with the ambience of {}',
    'A photo with an ambience of {}',
    'A photo with an {} ambience',
    '{} ambience photo',
    
    'A photo with the air of {}',
    'A photo with an air of {}',
    'A photo with an {} air',
    '{} air photo',
]

per_img_token_list = ['!','@','#','$','%','^','&','(',')']

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 specific_token = None,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 initializer_words = None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.specific_token = specific_token
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.initializer_words = initializer_words

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images <= len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])
        name = self.image_paths[i % self.num_images].split('/')[-1].split('.')[0]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
            template_p = random.random()
            template_order = 0 if template_p > 0.5 else 1 # tplt_idx 필요없어서 임의로 집어넣은 값 
            #template_p = random.random()
            """
            templates = [template_layout, imagenet_templates_small, template_texture]
            template_prob = [0.2, 0.8, 1.0]
            for i in range(len(templates)):
                if template_p <= template_prob[i]:
                    text = random.choice(templates[i]).format(placeholder_string)
                    template_order = np.array(i).astype(np.uint8)
                    break
            
            """
            """
            # concept-level decomposition
            templates = [template_layout, template_texture]
            for i in range(len(templates)):
                if template_p <= (i+1)/len(templates):
                    text = random.choice(templates[i]).format(placeholder_string)
                    template_order = np.array(i).astype(np.uint8)
                    break
            """
                
            """
            if template_p > 0.5:
                text = random.choice(template_layout).format(placeholder_string)
                template_order = np.array(0).astype(np.uint8)
            else:
                text = random.choice(template_texture).format(placeholder_string)
                template_order = np.array(1).astype(np.uint8)
            """
        example["template_idx"] = template_order
        #example["template_idx"] = i % (self.num_images * 2) # 0, 1, 2, 3, 4, 5, 6, 7 ########## Identity Preservation ##########
        example["caption"] = text
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

