import os
import torch
from io import BytesIO
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline
import matplotlib.pyplot as plt
from PIL import Image
import cv2

st.set_page_config(page_title='ControlNet', page_icon=Image.open(os.path.join("imgs","logo_tasti_light.png")), layout="centered", initial_sidebar_state="auto")

@st.cache_resource
def create_pipeline():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None)
    #controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    #pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_data
def create_image(_pipe, prompt, mask_image, num_inference_steps, guidance_scale, controlnet_conditioning_scale ):
    return _pipe(prompt=prompt, image=mask_image, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, controlnet_conditioning_scale  = controlnet_conditioning_scale).images[0]
# streamlit code

st.title("ControlNet with Latent Diffusion")

# a short paragraph explaining the models
st.markdown(
'''This is a ControlNet for Text-to-Image Generation with additional guidance provided by a scribble. For the purpose of this demo, we leveraged the model made available by the authors of the paper.  

The model takes in a prompt and a scribble, which are both used to guide the denoising diffusion process.  

The guidance scale controls the amount of influence the prompt has on the generated image and the continioning scale indicates the influence the conditioning image has on the process.  

The number of inference steps controls the number of steps the model takes to generate the image.'''
)

pipe = create_pipeline()

# create drawable canvas with the init image in the background that we can draw on to create the mask, fill color is white, background color is black
st.markdown("## Make a scribble")
st.markdown("Draw a large mask on the areas you want to inpaint.")
mask_image = st_canvas(drawing_mode="freedraw", key="canvas", fill_color="rgb(0, 0, 0)", stroke_width=10, stroke_color="rgb(255, 255, 255)", update_streamlit=True, height=700, width=700)
prompt = st.text_input("Prompt", "A car driving in the middle of the desert", help="The prompt to guide the generation of the image, i.e., what you want the model to generate")
num_inference_steps = st.slider("Number of Inference Steps", 10, 100, 20, help="The number of steps the model takes to generate the image. Higher values result in better quality images but take longer to generate.")
guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 7.5, 0.1, help="The amount of influence the prompt has on the generated image. Higher values result in images that more closely resemble the prompt but can be less diverse.")
controlnet_conditioning_scale  = st.slider("Conditioning Scale", 0.0, 2.0, 1.0, 0.1, help="The amount of influence the scribble has on the generated image. Higher values result in images that more closely resemble the scribble but can be less diverse.")

if st.button("Generate"):
    control_image = mask_image.image_data.copy()
    control_image = control_image[:,:,:-1]
    # resize the control image to 512x512
    control_image = cv2.resize(control_image, (512, 512))
    control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2GRAY)
    control_image[control_image > 10] = 255
    control_image[control_image <= 10] = 0
    control_image = Image.fromarray(control_image).convert("L")
    #st.image(control_image, caption="Generated Mask", use_column_width=True)
    image = create_image(pipe, prompt, control_image, num_inference_steps, guidance_scale, controlnet_conditioning_scale)
    st.image(image, caption="Generated Image", use_column_width=True)