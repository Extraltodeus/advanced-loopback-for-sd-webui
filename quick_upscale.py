
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
    def title(self):
        return "Lanczos quick upscale"

    def ui(self, is_img2img):
        upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Upscale factor', value=2)
        return [upscale_factor]

    def run(self, p, upscale_factor):
        infotexts = []
        def simple_upscale(img, upscale_factor):
            w, h = img.size
            w = int(w * upscale_factor)
            h = int(h * upscale_factor)
            return img.resize((w, h), Image.LANCZOS)

        state.job_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_samples = True
        output_images = []
        for batch_no in range(state.job_count):
            print(f"\nJob : {state.job_count}/{batch_no}\nSeed : {p.seed}\nPrompt : {p.prompt}\n")
            proc = process_images(p)
            infotexts.append(proc.info)
            proc.images[0] = simple_upscale(proc.images[0], upscale_factor)
            images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, proc.prompt, opts.samples_format, info= proc.info, p=p)
            output_images += proc.images
            p.seed = proc.seed + 1

        return Processed(p, images, infotexts=infotexts,index_of_first_image=0)
