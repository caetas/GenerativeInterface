import os
import torch
from io import BytesIO
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# set the page config, white page with centered layout and auto sidebar
st.set_page_config(page_title='Inpainting', page_icon=Image.open(os.path.join("imgs","logo_tasti_light.png")), layout="centered", initial_sidebar_state="auto")

# change background color of the page to white
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        background-color: #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def create_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", safety_checker = None,  torch_dtype = torch.float16).to("cuda")
    # pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker = None, torch_dtype = torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_sequential_cpu_offload()
    return pipe

@st.cache_data
def create_image(_pipe, prompt,negative_prompt, _init_image, mask_image, num_inference_steps, guidance_scale, strenght):
    return _pipe(prompt=prompt, negative_prompt = negative_prompt, image=_init_image, mask_image=mask_image, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, strenght = strenght).images[0]

@st.cache_data
def load_base_imgs():
    imgs = []
    prompts = ["Man with suit and tie, high resolution, standing in front of a building", "Face of a Brown and white cat, high resolution, sitting on a park bench", "A beautiful castle, high resolution", "An old pirate boat navigating in the sea"]
    imgs_path = os.listdir("imgs/inpainting")
    imgs_path.sort()
    for img_path in imgs_path:
        imgs.append(f"imgs/inpainting/{img_path}")
    return imgs, prompts

# streamlit code
st.title("Latent Diffusion Inpainting")

# a short paragraph explaining the models
st.markdown(
'''This is a Latent Diffusion model for Text-to-Image Inpainting. For the purpose of this demo, we leveraged the Stable Diffusion 2 Inpainting model made available by StabilityAI.  

The model takes in a prompt and a mask, which are both used to guide the denoising diffusion process.  

The guidance scale controls the amount of influence the prompt has on the generated image and the strength indicates the extent to transform the reference image.  

The number of inference steps controls the number of steps the model takes to generate the image.'''
)

pipe = create_pipeline()
imgs, prompts = load_base_imgs()

# upload an image as well
st.markdown("## Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
# if the user uploads an image, save it to the imgs list and prompts list
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    imgs.append(uploaded_file)
    prompts.append("Insert your own prompt!")

# select the image to inpaint and return the index of the selected image, allow for multiple files to be uploaded
init_image_path = st.selectbox("Select an Image", imgs, format_func=lambda x: f"Uploaded image {imgs.index(x)}" if imgs.index(x) > 3 else f"Image {imgs.index(x)}")
# if init is in imgs then the prompt is the corresponding prompt, otherwise it's a default prompt
initial_prompt = prompts[imgs.index(init_image_path)] if init_image_path in imgs else "Insert your own prompt!"

init_image = Image.open(init_image_path).resize((512, 512))
if init_image.mode != "RGB":
    init_image = init_image.convert("RGB")

# create drawable canvas with the init image in the background that we can draw on to create the mask, fill color is white, background color is black
st.text("Draw on the image to create the mask")
mask_image = st_canvas(background_image=init_image, drawing_mode="freedraw", key="canvas", fill_color="rgb(0, 0, 0)", stroke_width=40, stroke_color="rgb(255, 255, 255)", update_streamlit=True, height=700, width=700)
prompt = st.text_input("Prompt", initial_prompt, help="The prompt to guide the generation of the image, i.e., what you want the model to generate")
negative_prompt = st.text_input("Negative Prompt", "synthetic looking", help="The negative prompt to guide the generation of the image, i.e., what you don't want the model to generate")
num_inference_steps = st.slider("Number of Inference Steps", 10, 100, 50, help="The number of steps the model takes to generate the image. Higher values result in better quality images but take longer to generate.")
guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 9.5, 0.1, help="The amount of influence the prompt has on the generated image. Higher values result in images that more closely resemble the prompt but can be less diverse.")
strenght = st.slider("Strength", 0.0, 1.0, 1.0, 0.1, help="The extent to transform the reference image. When strength is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in num_inference_steps")

if st.button("Generate"):


    control_image = mask_image.image_data.copy()
    control_image = control_image[:,:,:-1]
    # resize the control image to 512x512
    control_image = cv2.resize(control_image, (512, 512))
    control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2GRAY)
    control_image[control_image > 10] = 255
    control_image[control_image <= 10] = 0
    control_image = Image.fromarray(control_image).convert("L")

    if negative_prompt == "":
        negative_prompt = None
    image = create_image(pipe, prompt, negative_prompt, init_image, control_image, num_inference_steps, guidance_scale, strenght)

    st.image(image, caption="Generated Image", use_column_width=True)