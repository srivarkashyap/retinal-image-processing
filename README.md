# Diabetic Retinopathy Diagnosis using Image processing and Deep Learning Methods
## Execution of the project
* Apply CLAHE Algorithm on the fundus images
* Augment the images using GAN image augmentation techniques
* Apply the DenseNet-201 for classification on the augmented CLAHE images
#### CLAHE Algorithm on the fundus images
* CLAHE – Contrast Limited Adaptive Histogram Equalization is an Image Equalization
method. CLAHE is a variation of Adaptive histogram equalization (AHE) that prevents contrast
over-amplification
* CLAHE works on small areas of an image called tiles rather than the complete image. The
surrounding tiles are blended using bilinear interpolation to remove the false boundaries. This
algorithm can be used to improve image contrast.
#### GAN image augmentation techniques
 * Data augmentation plays a crucial role in enhancing the performance of deep
learning models .
 * GANs offer a more advanced approach by generating realistic synthetic samples. By incorporating GAN-based augmentation, the dataset is expanded, enabling the model to generalize better and achieve improved performance in deep learning tasks.
#### DenseNet-201 for classification on the augmented CLAHE images
 * Dense connections: DenseNet-201 utilizes dense connections between layers, allowing
information to flow directly from earlier layers to later layers. This helps in preserving and reusing
features at different depths, which can be crucial for detecting intricate patterns associated with
DR.
* Feature extraction: DenseNet-201 has a deep architecture with multiple convolutional
layers. These layers are capable of automatically learning hierarchical representations and
extracting discriminative features from the retinal images, which can be indicative of DR-related
abnormalities.
* Performance: DenseNet-201 has a large number of parameters, which allows it to
capture intricate details in the images. This increased capacity can lead to higher accuracy in
detecting DR compared to models with fewer parameters

## Software Requirements:
Platform – Google Colab, Jupiter Notebook, Visual Studio<br>
Development Tool - CNN, Open CV<br>
Back End - Python<br>
