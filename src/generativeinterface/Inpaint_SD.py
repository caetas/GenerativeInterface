import os
import torch
from io import BytesIO
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title='Inpainting', page_icon=Image.open(os.path.join("imgs","logo_tasti_light.png")), layout="centered", initial_sidebar_state="auto")

@st.cache_resource
def create_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker = None)
    #pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker = None, torch_dtype = torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_data
def create_image(_pipe, prompt, _init_image, mask_image, num_inference_steps, guidance_scale, strenght):
    return _pipe(prompt=prompt, image=_init_image, mask_image=mask_image, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, strenght = strenght).images[0]

@st.cache_data
def load_base_imgs():
    imgs = []
    prompts = ["Man with suit and tie, high resolution, standing in front of a building", "Face of a Brown and white cat, high resolution, sitting on a park bench", "A beautiful castle, high resolution", "An old pirate boat navigating in the sea"]
    imgs_path = os.listdir("imgs/inpainting")
    imgs_path.sort()
    for img_path in imgs_path:
        imgs.append(Image.open(f"imgs/inpainting/{img_path}"))
    return imgs, prompts

# streamlit code

st.title("Latent Diffusion Inpainting")

# a short paragraph explaining the models
st.markdown(
'''This is a Latent Diffusion model for Text-to-Image Inpainting. For the purpose of this demo, we leveraged the Stable Diffusion Inpainting model made available by RunwayML.  

The model takes in a prompt and a mask, which are both used to guide the denoising diffusion process.  

The guidance scale controls the amount of influence the prompt has on the generated image and the strength indicates the extent to transform the reference image.  

The number of inference steps controls the number of steps the model takes to generate the image.'''
)

pipe = create_pipeline()
imgs, prompts = load_base_imgs()

# select the images and name each as Image 1, Image 2, etc
init_image = st.selectbox("Select Image", imgs, format_func=lambda x: "Image " + str(imgs.index(x) + 1))
initial_prompt = prompts[imgs.index(init_image)]
init_image = init_image.resize((512, 512))

# create drawable canvas with the init image in the background that we can draw on to create the mask, fill color is white, background color is black
st.text("Draw on the image to create the mask")
mask_image = st_canvas(background_image=init_image, drawing_mode="freedraw", key="canvas", fill_color="rgb(0, 0, 0)", stroke_width=20, stroke_color="rgb(255, 255, 255)", update_streamlit=True, height=700, width=700)
prompt = st.text_input("Prompt", initial_prompt, help="The prompt to guide the generation of the image, i.e., what you want the model to generate")
num_inference_steps = st.slider("Number of Inference Steps", 10, 100, 20, help="The number of steps the model takes to generate the image. Higher values result in better quality images but take longer to generate.")
guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 7.5, 0.1, help="The amount of influence the prompt has on the generated image. Higher values result in images that more closely resemble the prompt but can be less diverse.")
strenght = st.slider("Strength", 0.0, 1.0, 0.5, 0.1, help="The extent to transform the reference image. When strength is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in num_inference_steps")

if st.button("Generate"):
    control_image = mask_image.image_data.copy()
    control_image = Image.fromarray(control_image).resize((512, 512))
    image = create_image(pipe, prompt, init_image, control_image, num_inference_steps, guidance_scale, strenght)
    st.image(image, caption="Generated Image", use_column_width=True)