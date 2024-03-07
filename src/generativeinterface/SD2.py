import PIL
import requests
import torch
from io import BytesIO
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from PIL import Image

@st.cache_resource
def create_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", safety_checker = None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_data
def create_image(_pipe, prompt, init_image, mask_image, num_inference_steps):
    return _pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps = num_inference_steps).images[0]

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

# streamlit code

st.title("Stable Diffusion Inpainting")

pipe = create_pipeline()

#init_image = st.file_uploader("Initial Image", type=["png", "jpg", "jpeg"])
#mask_image = st.file_uploader("Mask Image", type=["png", "jpg", "jpeg"])
init_image = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
#mask_image = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# if both images are uploaded then we can input the prompt
if init_image:
    init_image = download_image(init_image).resize((512, 512))
    # create drawable canvas with the init image in the background that we can draw on to create the mask, fill color is white, background color is black
    mask_image = st_canvas(background_image=init_image, drawing_mode="freedraw", key="canvas", fill_color="rgb(0, 0, 0)", stroke_width=20, stroke_color="rgb(255, 255, 255)", update_streamlit=True, height=700, width=700)
    #mask_image = download_image(mask_image).resize((512, 512))
    # show both images
    #st.image([init_image, mask_image])
    num_inference_steps = st.slider("Number of Inference Steps", 1, 100, 10)
    prompt = st.text_input("Prompt", "face of a Brown and white cat, high resolution, sitting on a park bench")
    if st.button("Generate"):
        control_image = mask_image.image_data.copy()
        control_image = Image.fromarray(control_image).resize((512, 512))
        image = create_image(pipe, prompt, init_image, control_image, num_inference_steps)
        st.image(image, caption="Generated Image", use_column_width=True)