import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
import joblib

def load_model(fine_tuned_model_path):
    # Load the base model with VGGFace2 pretrained weights
    model = InceptionResnetV1(pretrained='vggface2')

    # Modify the model architecture to match the fine-tuned model
    num_classes = 49  # Assuming the fine-tuned model has 49 output classes
    in_features = model.logits.in_features
    model.logits = nn.Linear(in_features, num_classes)

    # Load the fine-tuned model weights
    state_dict = torch.load(fine_tuned_model_path)

    # Adjust the state_dict to match the modified model architecture
    state_dict['logits.weight'] = state_dict['logits.weight'][:, :in_features]
    state_dict['logits.bias'] = state_dict['logits.bias'][:num_classes]

    # Load the modified state_dict
    model.load_state_dict(state_dict)
    return model

# Data directory
data_dir = 'cropped_faces'

# Define image transformations
trans = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load data
dataset = datasets.ImageFolder(data_dir, transform=trans)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # use vggface2 model
model = InceptionResnetV1(pretrained='vggface2')

# # use fine tuned model
# model = load_model("model/resnet_model.pth")

#Set the model to evaluation mode
model.eval().to(device)

# Extract features
embeddings = []
labels = []

with torch.no_grad():
    for images, label in data_loader:
        images = images.to(device)
        embeddings.append(model(images).cpu())
        labels.append(label)

# Concatenate features and labels
embeddings = torch.cat(embeddings)
labels = torch.cat(labels)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(embeddings, labels)

# Save the model and class names
joblib.dump((svm_classifier, dataset.classes), 'model/face_classifier.joblib')
print("Model and class names saved as face_classifier.joblib")
