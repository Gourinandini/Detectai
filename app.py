import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import timm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class AIDetector(nn.Module):
    def __init__(self):
        super(AIDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.backbone(x)

# Load model
model = AIDetector().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def detect(image):
    img    = Image.fromarray(image).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob = model(tensor).item()
    
    if prob > 0.5:
        label      = "✅ REAL IMAGE"
        confidence = prob
    else:
        label      = "🤖 AI GENERATED"
        confidence = 1 - prob
    
    return f"{label}\nConfidence: {confidence*100:.1f}%"

# Gradio Interface
interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Textbox(label="Result"),
    title="🔍 AI Image Detector",
    description="Upload an image to check if it is Real or AI Generated. Note: Model is trained on CIFAR style images."
)

interface.launch()