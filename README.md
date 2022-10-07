# advanced-loopback-for-sd-webui

This script is made to be used with the [AUTOMATIC1111 webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Installation

Drop the script into your /scripts folder.

Use the --allow-code argument.

If you're using a Google Colab, you can add this in a code block before the one that starts the webui :

    !wget https://raw.githubusercontent.com/Extraltodeus/advanced-loopback-for-sd-webui/main/advanced_loopback.py -O /content/stable-diffusion-webui/scripts/advanced_loopback.py

## Features

I will only describe the less obvious feats since you don't need to be explained what a zoom is :)

The scripts UI looks like this and will show up in your img2img tab :

![image](https://user-images.githubusercontent.com/15731540/194636007-4dfaa7a1-b8d2-48a5-bd8d-b88b69fb8c90.png)

- Use first image colors (custom color correction)

This feature will use the colors from the init image instead of the last generated one (like the one already availaible does). This allows to avoid loosing colors when using the zoom feature.

- Direction X/Y

Will shift the cropped/zoomed next image up/down/left/right. The value is limited by the zoom level.

- Denoising strength start/end

![image](https://user-images.githubusercontent.com/15731540/194637537-8b5dcb0e-f3c7-45bd-9c09-e60ffa0b76a8.png)

This allows for a progressive parameter change (or sinusoidal, read below). Without the sine option enabled, it will simply put in proportion the starting value slider with the ending value.

For example if you set it like this :

![image](https://user-images.githubusercontent.com/15731540/194638100-56389bf2-fd81-4721-801e-36df59414aba.png)

The 5 first generated images will have a value of 0.2, then from 6 to 10 will ramp up from 0.2 to 0.6.

This can be nice the have some visual rest in between the transitions while zooming.

- Saturation enhancement per image

Will increase/decrease slightly the saturation from one image to the next.

If you are using the color correction, it will ony influence the contrasts.

- Sine/exponentiation

![image](https://user-images.githubusercontent.com/15731540/194638484-351a6401-51b3-4c5d-a705-cb4372c08e5e.png)

Allows a sinusoidal variation of these two parameters (the widest curve in the image below).

Both of the options uses the start/end input boxes above. The maximum strength will be in between the minimum and maximum value and will then loop.

The exponentiation sliders on the right kmakes the curves tighter, which will make the changes more sudden.

![expcurves](https://user-images.githubusercontent.com/15731540/194625416-ca0f3a3d-f0a3-4d00-9146-f30ac508a46b.png)

The phase difference allows you to have the two variations not synchronised

![phase](https://user-images.githubusercontent.com/15731540/194625410-97754da7-4e61-49d9-9305-eeea1ec712cf.png)

- prompt/seed options

![image](https://user-images.githubusercontent.com/15731540/194639114-86b477f3-8d94-4e7f-975a-828d93d2f738.png)

The multiple prompts option will switch prompt after the "end" imagee has been reached (refering to the input box).

So 0-10 will switch prompt every 10 images.

The prompts have to be negative/positive, one per line. Just leave a blank line if you do not wish to use a negative prompt.

![image](https://user-images.githubusercontent.com/15731540/194639713-80063fd9-cb55-40a0-a588-28cd78ac164f.png)

Note : "Same seed for everything" will accentuate the samplers noise. Euler A with the zoom will mostly create "warp drive" effects like in the example below.

## Example

Unedited output with prompt switching/zoom/sine variations/same seed per prompt.

"What you will see is 100% artificial" and birth to death related prompts.

https://user-images.githubusercontent.com/15731540/194624678-544e52db-88d1-429e-b4f7-2a2a1e35cc60.mp4


### End note

For now the script does not output a video immediatly. Just the images. I personnaly use ffmpeg to put it all together.
