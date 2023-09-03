import torch
from torchvision import transforms
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained image feature extractor (e.g., ResNet)
image_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
image_model.eval()

# Load pre-trained language model (e.g., GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Load and preprocess an image
image_path = "image.jpg"
image = Image.open(image_path)
preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
image = preprocess(image).unsqueeze(0)  # Add batch dimension

# Extract image features
with torch.no_grad():
    image_features = image_model(image)

# Generate captions
input_text = "A picture of "  # You can start with any initial text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the generated caption
generated_caption = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", generated_caption)
