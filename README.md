<p align="center">
  <img src="data/raw/logo_tasti_light.png" width="70%" alt='Tasti Project'>
</p>

# Generative Interface

[![Python](https://img.shields.io/badge/python-3.9+-informational.svg)](https://www.python.org/downloads/release/python-3918/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=black)](https://pycqa.github.io/isort)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![wandb](https://img.shields.io/badge/tracking-wandb-blue)](https://wandb.ai/site)
[![dvc](https://img.shields.io/badge/data-dvc-9cf)](https://dvc.org)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

Interface to interact with several generative models.

**This work is part of the Xecs TASTI project, nr. 2022005.**

## Prerequisites

You will need:

- `python` (see `pyproject.toml` for full version)
- `Git`
- `Make`
- a `.secrets` file with the required secrets and credentials
- load environment variables from `.env`

## Installation

Clone this repository (requires git ssh keys)

    git clone --recursive git@github.com:caetas/GenerativeInterface.git
    cd generativeinterface

Install dependencies

    conda create -y -n python3.9 python=3.9
    conda activate python3.9

or if environment already exists

    conda env create -f environment.yml
    conda activate python3.9

### On Linux

And then setup all virtualenv using make file recipe

    (python3.9) $ make setup-all

You might be required to run the following command once to setup the automatic activation of the conda environment and the virtualenv:

    direnv allow

Feel free to edit the [`.envrc`](.envrc) file if you prefer to activate the environments manually.

### On Windows

You can setup the virtualenv by running the following commands:

    python -m venv .venv-dev
    .venv-dev/Scripts/Activate.ps1
    python -m pip install --upgrade pip setuptools
    pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements/requirements-win.txt

To run the code please remember to always activate both environments:

    conda activate python3.9
    .venv-dev/Scripts/Activate.ps1

## Run the Interfaces

### Text-to-Image Latent Diffusion

If you are using a Nvidia GPU that supports `fp16` operations, make sure that the code in [`SD2.py`](src/generativeinterface/SD2.py) looks like this:

```python
def create_pipeline():
    #pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker = None)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker = None, torch_dtype = torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe
```

If you have more than 6GB of VRAM available, you can comment the following line as well:

```python
pipe.enable_sequential_cpu_offload()
```

To create the interface, please run the following commands:

    cd src/generativeinterface
    streamlit run SD2.py

### Text-To-Image Inpainting

If you are using a Nvidia GPU that supports `fp16` operations, make sure that the code in [`Inpaint_SD.py`](src/generativeinterface/Inpaint_SD.py) looks like this:

```python
def create_pipeline():
    #pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker = None)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", safety_checker = None, torch_dtype = torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe
```

If you have more than 6GB of VRAM available, you can comment the following line as well:

```python
pipe.enable_sequential_cpu_offload()
```

To create the interface, please run the following commands:

    cd src/generativeinterface
    streamlit run Inpaint_SD.py

### ControlNet

If you are using a Nvidia GPU that supports `fp16` operations, make sure that the code in [`Control_SD.py`](src/generativeinterface/Control_SD.py) looks like this:

```python
def create_pipeline():
    #controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    #pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_sequential_cpu_offload()
    return pipe
```

If you have more than 6GB of VRAM available, you can comment the following line as well:

```python
pipe.enable_sequential_cpu_offload()
```

To create the interface, please run the following commands:

    cd src/generativeinterface
    streamlit run Control_SD.py

## Documentation

Full documentation is available here: [`docs/`](docs).

## Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information.

## Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md]) for details and
the process for submitting pull requests to us.

## License

This project is licensed under the terms of the `MIT` license.
See [LICENSE](LICENSE) for more details.

## References

This work is based on the tutorials and documentation provided by [HuggingFace](https://huggingface.co/) and the [Diffusers](https://github.com/huggingface/diffusers) library:

- [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)