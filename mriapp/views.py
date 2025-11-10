import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.urls import reverse
from .forms import UploadImageForm
from .utils import run_alexnet, run_local_model

def index(request):
    return render(request, 'upload.html', {})

def upload_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            save_path = os.path.join(uploads_dir, image.name)
            with open(save_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Run both models
            alex = run_alexnet(save_path)
            local = run_local_model(save_path)

            # prepare comparison
            comparison = {
                'alex_label': alex['label'],
                'alex_score': alex['score'],
                'local_label': local['label'],
                'local_score': local['score'],
                'agree': alex['label'] == local['label']
            }

            # Build context and render result page
            context = {
                'image_url': settings.MEDIA_URL + 'uploads/' + image.name,
                'comparison': comparison,
                'alex': alex,
                'local': local,
            }
            return render(request, 'result.html', context)
    else:
        form = UploadImageForm()
    return render(request, 'upload.html', {'form': form})
