import os
from PIL import Image
import numpy as np

def run_local_model(image_path):
    """Simulated local model: simple heuristic based on mean intensity.

    Returns a dict: {'label': 'tumor'|'no tumor', 'score': float}
    """
    img = Image.open(image_path).convert('L')  # grayscale
    arr = np.array(img).astype(float)
    mean = arr.mean() / 255.0
    # heuristic: darker images -> tumor (arbitrary for demo)
    score = 1.0 - mean  # higher means more likely tumor
    label = 'tumor' if score >= 0.5 else 'no tumor'
    return {'label': label, 'score': float(score)}

def run_alexnet(image_path):
    """Try to run AlexNet pretrained from torchvision.
    If torch is not available or any error occurs, fall back to a simulated result.
    Returns a dict: {'label': 'tumor'|'no tumor', 'score': float, 'top1': str}
    """
    try:
        import torch
        from torchvision import models, transforms
        from PIL import Image
        import torch.nn.functional as F

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.alexnet(pretrained=True).to(device)
        model.eval()

        input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_tensor)
            probs = F.softmax(out, dim=1)
            top1_prob, top1_catid = torch.max(probs, dim=1)
            top1_prob = float(top1_prob.item())
            top1_catid = int(top1_catid.item())

        # For demo: map top1 probability to a binary decision (arbitrary)
        label = 'tumor' if top1_prob >= 0.5 else 'no tumor'
        return {'label': label, 'score': top1_prob, 'top1': str(top1_catid)}
    except Exception as e:
        # fallback simulation: small random-ish deterministic function from file path
        import hashlib
        h = hashlib.md5(image_path.encode()).hexdigest()
        val = int(h[:8], 16) % 100 / 100.0
        label = 'tumor' if val >= 0.5 else 'no tumor'
        return {'label': label, 'score': float(val), 'top1': 'simulated'}
