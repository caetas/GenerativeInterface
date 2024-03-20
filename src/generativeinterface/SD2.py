import torch
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title='Stable Diffusion 2', page_icon=Image.open(os.path.join("imgs","logo_tasti_light.png")), layout="centered", initial_sidebar_state="auto")

@st.cache_resource
def create_pipeline():
    #pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker = None)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker = None, torch_dtype = torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_data
def create_image(_pipe, prompt, num_inference_steps, guidance_scale):
    return _pipe(prompt=prompt, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale).images[0]

# streamlit code

st.title("Text-to-Image with Latent Diffusion")

# a short paragraph explaining the models
st.markdown(
'''This is a Latent Diffusion model for Text-to-Image Generation. For the purpose of this demo, we leveraged the Stable Diffusion 2.1 model made available by StabilityAI.  

The model takes in a prompt, which is used to guide the denoising diffusion process.  

The guidance scale controls the amount of influence the prompt has on the generated image.  

The number of inference steps controls the number of steps the model takes to generate the image.'''
)

pipe = create_pipeline()

prompt = st.text_input("Prompt", "A Brown and white cat, high resolution", help="The prompt to guide the generation of the image, i.e., what you want the model to generate")
num_inference_steps = st.slider("Number of Inference Steps", 5, 100, 10, help="The number of steps the model takes to generate the image. Higher values result in better quality images but take longer to generate.")
guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 7.5, 0.1, help="The amount of influence the prompt has on the generated image. Higher values result in images that more closely resemble the prompt but can be less diverse.")
if st.button("Generate"):
    image = create_image(pipe, prompt, num_inference_steps, guidance_scale)
    st.image(image, caption="Generated Image", use_column_width=True)