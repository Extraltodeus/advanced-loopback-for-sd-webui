
import math
import os
import sys
import traceback
import random

import modules.scripts as scripts
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    alwayson = True
    def title(self):
        return "Lanczos simple upscale"
      
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        simple_upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Upscale factor', value=2)
        return [simple_upscale_factor]
      
    def batch_postprocess(self, p, image, simple_upscale_factor, **batch_properties):
        w, h = image.size
        w = int(w * simple_upscale_factor)
        h = int(h * simple_upscale_factor)
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        images.save_image(image, p.outpath_samples, "", batch_properties['seed'], batch_properties['prompt'], opts.samples_format, info=batch_properties['info'], p=p)
