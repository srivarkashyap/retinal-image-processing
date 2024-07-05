import streamlit as st
from PIL import Image

# Streamlit app title
st.title("Image Upload and Description")

# Upload an image
image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image is not None:
    # Display the uploaded image on the initial page
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a button to move to the new page
    if st.button("View Image Description"):
        # Move to the new page
        st.text("")  # Add some space
        st.header("Image Description")

        # Add a text input for image description
        description = st.text_area("Add a description for the image", "")

        # Display the image with the provided description on the "new" page
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Image Description:", description)
