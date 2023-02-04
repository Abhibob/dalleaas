#!/usr/bin/env python
DALLE_COMMIT_ID = None

DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

import shutil
from time import time
from uuid import uuid4
import jax
import jax.numpy as jnp
import random
import base64

jax.local_device_count()

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from functools import partial
from io import BytesIO

model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route("/generate-images")
def generate():
    prompts = request.get_json()["prompts"]
    n_predictions = request.get_json()["image_number"]
    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    print(f"Prompts: {prompts}\n")
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))

        target = '/tmp/apiimages/' + str(uuid4()) + "/"
        os.mkdir(target)

        for decoded_img in range(len(decoded_images)):
            img = Image.fromarray(np.asarray(decoded_images[decoded_img] * 255, dtype=np.uint8))
            output = BytesIO()
            img.save(output, format="JPEG")
            image_data = base64.b64encode(output.getvalue())
            if not isinstance(image_data, str):
                image_data = image_data.decode()

            open(target + "image_" + str(decoded_img + 1) + ".png", "w").write("data:image/jpg;base64," + image_data)

            # img.show()

        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            for file in glob(os.path.join(target, '*.png')):
                zf.write(file, os.path.basename(file))
        stream.seek(0)

        shutil.rmtree(target)

        return send_file(
            stream,
            as_attachment=True,
            attachment_filename='archive.zip'
        )