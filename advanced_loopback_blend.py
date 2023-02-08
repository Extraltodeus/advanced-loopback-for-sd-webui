import numpy as np
from tqdm import trange
from PIL import Image, ImageEnhance

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from copy import deepcopy
from math import sin, pi

class Script(scripts.Script):
    def title(self):
        return "Advanced loopback blend"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        loops = gr.Number(minimum=1, step=1, label='Loops', value=4)
        use_first_image_colors  = gr.Checkbox(label='Use first image colors (custom color correction)     ', value=False)
        denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor (overridden if proportional used)', value=1)
        with gr.Row():
            zoom_level = gr.Slider(minimum=0, maximum=50, step=1, label='Zoom level     ', value=0)
            zoom_blend = gr.Checkbox(label='Blend 50/50 with original when zoomed. Doesn\'t work with sine variation.', value=False)
            # zoom_refresh = gr.Slider(minimum=0, maximum=50, step=1, label='Refresh base image for blend every n iterations', value=0)
        #     direction_x = gr.Slider(minimum=-0.1, maximum=0.1, step=0.01, label='Direction X', value=0)
        #     direction_y = gr.Slider(minimum=-0.1, maximum=0.1, step=0.01, label='Direction Y', value=0)
        with gr.Row():
            denoising_strength_first_image = gr.Number(minimum=0, step=1, label='Denoising strength start   ', value=0)
            denoising_strength_last_image  = gr.Number(minimum=0, step=1, label='Denoising strength end   ',   value=4)
        denoising_strength_min  = gr.Slider(minimum=0.1, maximum=1, step=0.01, label='Denoising strength proportional change starting value   ', value=0.1)
        denoising_strength_max  = gr.Slider(minimum=0.1, maximum=1, step=0.01, label='Denoising strength proportional change ending value (0.1 = disabled)   ', value=0.1)
        cfg_scale_min  = gr.Slider(minimum=0.1, maximum=30, step=0.1, label='CFG scale proportional change starting value   ', value=0.1)
        cfg_scale_max  = gr.Slider(minimum=0.1, maximum=30, step=0.1, label='CFG scale proportional change ending value (0.1 = disabled)   ', value=0.1)
        saturation_per_image    = gr.Slider(minimum=0.99, maximum=1.01, step=0.001, label='Saturation enhancement per image   ', value=1)
        with gr.Row():
            use_sine_variation_dns  = gr.Checkbox(label='Use sine denoising strength variation (CFG will be scaled with it if the slider is > 0.1)',      value=True)
            phase_diff_denoising    = gr.Slider(minimum=0, maximum=1, step=0.05, label='Phase difference', value=0)
            amplify_sine_variation_denoise = gr.Slider(minimum=1, maximum=10, step=1, label='Denoising strength exponentiation     ', value=1)
        with gr.Row():
            use_sine_variation_zoom = gr.Checkbox(label='Use sine zoom variation', value=False)
            phase_diff_zoom = gr.Slider(minimum=0, maximum=1, step=0.05, label='Phase difference', value=0)
            amplify_sine_variation_zoom = gr.Slider(minimum=1, maximum=10, step=1, label='Zoom exponentiation     ', value=1)
        with gr.Row():
            use_multi_prompts    = gr.Checkbox(label='Use multiple prompts', value=False)
            same_seed_per_prompt = gr.Checkbox(label='Same seed per prompt', value=False)
            same_seed_always     = gr.Checkbox(label='Same seed for everything', value=False)
            same_init_image      = gr.Checkbox(label='Original init image for everything', value=False)
        multi_prompts     = gr.Textbox(label="Multiple prompts : 1 line positive, 1 line negative, leave a blank line for no negative", lines=2, max_lines=2000)
        return [
        loops,
        denoising_strength_change_factor,
        zoom_level,
        zoom_blend,
        # zoom_refresh,
        # direction_x,
        # direction_y,
        denoising_strength_first_image,
        denoising_strength_last_image,
        denoising_strength_min,
        denoising_strength_max,
        cfg_scale_min,
        cfg_scale_max,
        saturation_per_image,
        use_first_image_colors,
        use_sine_variation_dns,
        use_sine_variation_zoom,
        phase_diff_zoom,
        use_multi_prompts,
        multi_prompts,
        amplify_sine_variation_zoom,
        same_seed_per_prompt,
        phase_diff_denoising,
        amplify_sine_variation_denoise,
        same_seed_always,
        same_init_image
        ]

    def zoom_into(self, img, zoom):
        w, h = img.size
        img = img.crop((zoom,zoom,w-zoom,h-zoom))
        return img.resize((w, h), Image.LANCZOS)

    def run(self, p,
    loops,
    denoising_strength_change_factor,
    zoom_level,
    zoom_blend,
    # zoom_refresh,
    # direction_x,
    # direction_y,
    denoising_strength_first_image,
    denoising_strength_last_image,
    denoising_strength_min,
    denoising_strength_max,
    cfg_scale_min,
    cfg_scale_max,
    saturation_per_image,
    use_first_image_colors,
    use_sine_variation_dns,
    use_sine_variation_zoom,
    phase_diff_zoom,
    use_multi_prompts,
    multi_prompts,
    amplify_sine_variation_zoom,
    same_seed_per_prompt,
    phase_diff_denoising,
    amplify_sine_variation_denoise,
    same_seed_always,
    same_init_image
    ):

        ppos = []
        pneg = []
        if use_multi_prompts :
            prompts_list = multi_prompts.splitlines()
            oddeven = lambda x: 1 if x%2==0 else 0
            for x in range(len(prompts_list)) :
                if oddeven(x):
                    ppos.append(prompts_list[x])
                else:
                    pneg.append(prompts_list[x])
            if len(pneg) < len(ppos) :
                pneg.append("")

        def remap_range(value, minIn, MaxIn, minOut, maxOut):
            if value > MaxIn: value = MaxIn;
            if value < minIn: value = minIn;
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

        def get_sin_steps(i,amplify,phase_diff=0):
            i -= denoising_strength_first_image
            range = (denoising_strength_last_image - denoising_strength_first_image)
            x = i % (range)
            y = remap_range(x,0,range,0,1)
            y = y ** amplify
            z = sin((y+phase_diff/2)*pi)
            return z

        processing.fix_seed(p)
        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
            'Denoising strength proportional change start image':denoising_strength_first_image,
            'Denoising strength proportional change end image':denoising_strength_last_image,
            'Denoising strength proportional change starting value':denoising_strength_min,
            'Denoising strength proportional change ending value':denoising_strength_max,
            'CFG min':cfg_scale_min,
            'CFG max':cfg_scale_max,
            'use first image colors': use_first_image_colors,
            'Saturation enhancement per image':saturation_per_image,
            'Zoom level':zoom_level,
        }

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        state.job_count = loops * batch_count

        original_image = p.init_images[0].copy()
        original_image_for_zoom = p.init_images[0].copy()
        if opts.img2img_color_correction:
            p.color_corrections = [processing.setup_color_correction(p.init_images[0])]


        for n in range(batch_count):
            history = []
            multi_prompts_index = 0
            loops = round(loops)
            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True

                if use_multi_prompts :
                    image_range = (denoising_strength_last_image - denoising_strength_first_image)
                    il = i % (image_range)
                    if i == 0:
                        p.prompt = ppos[multi_prompts_index]
                        p.negative_prompt = pneg[multi_prompts_index]
                        print("Prompt :",p.prompt)
                        print("Negative prompt :",p.negative_prompt)
                    if il == 0 and i > 0:
                        multi_prompts_index+=1
                        try:
                            if same_seed_per_prompt:
                                if not same_seed_always:
                                    p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
                                    p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
                            p.prompt = ppos[multi_prompts_index]
                            p.negative_prompt = pneg[multi_prompts_index]
                        except Exception as e:
                            multi_prompts_index = 0
                            if same_seed_per_prompt:
                                if not same_seed_always:
                                    p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
                                    p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
                            p.prompt = ppos[multi_prompts_index]
                            p.negative_prompt = pneg[multi_prompts_index]
#                         print("Prompt :",p.prompt)
#                         print("Negative prompt :",p.negative_prompt)

                if use_first_image_colors:
                    p.color_corrections = [processing.setup_color_correction(original_image)]

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                if denoising_strength_max > 0.1 :
                    if use_sine_variation_dns :
                        ds = remap_range(get_sin_steps(i,amplify_sine_variation_denoise,phase_diff_denoising),0,1,denoising_strength_min,denoising_strength_max)
                    else:
                        ds = remap_range(i+1,denoising_strength_first_image,denoising_strength_last_image,denoising_strength_min,denoising_strength_max)
                    p.denoising_strength = round(ds,3)
                    print("Denoising strength : "+str(p.denoising_strength))

                if cfg_scale_max > 0.1 :
                    if use_sine_variation_dns :
                        cfgs = remap_range(get_sin_steps(i,amplify_sine_variation_denoise,phase_diff_denoising),0,1,cfg_scale_min,cfg_scale_max)
                    else:
                        cfgs = remap_range(i+1,denoising_strength_first_image,denoising_strength_last_image,cfg_scale_min,cfg_scale_max)
                    p.cfg_scale = round(cfgs,2)
                    print("CFG scale : "+str(p.cfg_scale))

                processed = processing.process_images(p)
                if zoom_level > 0:
                    if use_sine_variation_zoom :
                        if loops >= denoising_strength_first_image :
                            z = remap_range(get_sin_steps(i,amplify_sine_variation_zoom,phase_diff_zoom),0,1,1,zoom_level)
                            processed.images[0] = self.zoom_into(processed.images[0], z)
                            print("Zoom level :",z)
                    else:
                        processed.images[0] = self.zoom_into(processed.images[0], zoom_level)
                        if zoom_blend:
                            # if zoom_refresh > 0 and i%zoom_refresh == 0:
                            #     original_image_for_zoom = processed.images[0].copy()
                            original_image_zoomed = self.zoom_into(original_image_for_zoom.copy(), zoom_level*(i+1))
                            processed.images[0] = Image.blend(original_image_zoomed.copy().convert('RGB').resize(processed.images[0].size, Image.LANCZOS), processed.images[0].copy().convert('RGB'), alpha=0.5)


                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                if not same_init_image :
                    init_img = processed.images[0]
                else:
                    init_img = original_image

                if saturation_per_image != 1 :
                    init_img = ImageEnhance.Color(init_img).enhance(saturation_per_image)

                p.init_images = [init_img]
                if not same_seed_per_prompt:
                    if not same_seed_always:
                        p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
                        p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
                p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                history.append(processed.images[0])
                if state.interrupted:
                    break

            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            grids.append(grid)
            all_images += history

        if opts.return_grid:
            all_images = grids + all_images

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
