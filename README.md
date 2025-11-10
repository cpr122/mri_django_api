# MRI Django API (Upload + AlexNet comparison)

This project is a minimal Django application that allows:
- uploading an MRI image,
- running a prediction with **AlexNet (pretrained)** if PyTorch is installed,
- running a **simulated local model** (fallback) for comparison,
- returning a binary prediction ("tumor" / "no tumor") for both models,
- showing the uploaded image and a comparison page.

> **Note:** AlexNet (ImageNet) is not trained for medical images — when using a model
> trained on ImageNet the result is *not* clinically meaningful. This scaffold provides
> the wiring to perform inference; you should replace the simulated model with a
> properly trained model for MRI tumor detection.

## Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:
- Django
- Pillow
- torch
- torchvision

If you don't have CUDA or want to avoid installing torch, the app will fall back to a simulated model.

## Run

1. Apply migrations (the project only uses the default DB):
```bash
python manage.py migrate
```

2. Run the development server:
```bash
python manage.py runserver
```

3. Open http://127.0.0.1:8000/ to upload an image.

Uploaded files are stored in `media/uploads/`.

## Project structure

- `mri_project/` - Django project settings & URLs
- `mriapp/` - Django app with upload view and templates
- `media/` - runtime uploads (created when you upload)
- `requirements.txt` - python dependencies

## How inference works in this scaffold

- The code will try to import `torch` and `torchvision`:
  - If available, it will preprocess the image, load `alexnet(pretrained=True)`,
    run forward and compute a softmax; it derives a binary decision using a simple
    heuristic (top probability > 0.5 => "tumor").
  - If `torch` is not available, it will use a simulated model that classifies
    depending on the mean intensity of the grayscale image (simple heuristic).
- Replace `mriapp/utils.py` functions `run_alexnet` and `run_local_model` with
  your trained model for production use.

