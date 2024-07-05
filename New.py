import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np 

# Set the title and page icon
st.set_page_config(page_title="DR Detection", page_icon=":eyes:")

# Add a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
input_image1 = None

col1, col2 = st.columns(2)





with col1:
   st.header("Upload an Image")
   if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Input Image',width=300)

with col2:
     st.header("Result")
    
    #  st.write("Predicted Severity Level:", predicted_severity)
     if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        image_bw = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
        final_img = clahe.apply(gray_image)
        # st.image(image, caption="Uploaded image", use_column_width=True)
        # st.image(final_img, caption="After CLAHE", use_column_width=True)
        input_image1 = final_img

        # Load the pretrained DenseNet201 model
        model = models.densenet201(pretrained=False)
        num_classes = 4  # Number of classes for DR severity levels

        # Modify the classifier to match the shape of the loaded state_dict
        model.classifier = torch.nn.Linear(1920, num_classes)

        # Load the trained weights
        state_dict = torch.load('trained_model2.pth', map_location=torch.device('cpu'))
        state_dict['classifier.weight'] = state_dict['classifier.weight'][:num_classes, :]
        state_dict['classifier.bias'] = state_dict['classifier.bias'][:num_classes]
        model.load_state_dict(state_dict)

        # Set the model to evaluation mode
        model.eval()

        # Preprocess the input image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # Normalization parameters for the ImageNet dataset
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if input_image1 is not None:
            input_image = Image.fromarray(input_image1)
            input_image = input_image.convert("RGB")  # Convert to RGB mode

            # Preprocess the input image
            input_tensor = transform(input_image)
            input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

            # Perform the forward pass
            with torch.no_grad():
                # Make predictions
                output = model(input_tensor)

            # Get the predicted class index
            _, predicted_index = torch.max(output, 1)

            # Define severity level labels
            severity_levels = ["No DR", "Mild", "Moderate", "Severe","proliferate"]

            # Get the predicted severity level
            predicted_severity = severity_levels[predicted_index.item()]

                    
            # Show the predicted severity level
            st.write("Predicted Severity Level:", predicted_severity)





