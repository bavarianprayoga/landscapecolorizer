# **Landscape Colorizer**
A deep learning-based image colorization tool capable to colorize grayscale landscape images. Utilizing a ResNet-18 U-Net architecture, the model uses transfer learning to understand the image structure (trees, sky, mountains) and applies a regression approach to predict the a and b chrominance channels in the CIE Lab color space

![Output example](/outputs/plots/example_comparison.jpeg)

Special thanks to mberkay0 for the idea

https://github.com/mberkay0/image-colorization

## **Theory & Methodology**

### 1. CIE Lab Color Space 
Unlike the standard RGB color space, where color information is entangled with brightness across all three channels, this project utilizes CIE Lab:

- L (Lightness): Carries all luminosity and detail information (0-100).

- a & b (Color): Carry the Green-Red and Blue-Yellow color opponents.

The model is not tasked with generating the entire image from scratch. By feeding the grayscale image as the L-channel input, the model only needs to predict the missing a and b channels. This significantly simplifies the learning objective compared to predicting RGB values directly.

### 2. Encoder-Decoder Architecture (U-Net)
The core architecture is a U-Net variant, characterized by a contracting path (Encoder) and an expansive path (Decoder).

Encoder (ResNet-18 Backbone): Instead of training a feature extractor from scratch, we utilize a pre-trained ResNet-18 (weights from ImageNet). We truncate the classification head to extract deep latent representations.

Decoder (Upsampling): A mirrored upsampling + Conv2d that restores the image to 512×512 resolution.

### 3. Skip Connections
A critical component of the U-Net design is the use of Skip Connections.

The Problem: As images pass through the encoder (pooling layers), spatial information is compressed and lost. A standard decoder struggles to recover sharp edges (high-frequency details) from this compressed state.

The Solution: Concatenate feature maps from early ResNet layers directly to their corresponding layers in the Decoder. This injects high-resolution spatial information back into the generation process, ensuring that colors stay inside the lines of objects.

## Dataset
The model was fine-tuned on the Landscape Pictures dataset, comprising approximately 4,300 high-resolution images of nature scenes (Forests, Mountains, Deserts, Oceans).

https://www.kaggle.com/datasets/arnaud58/landscape-pictures

- landscapes (900 pictures)
- landscapes mountain (900 pictures)
- landscapes desert (100 pictures)
- landscapes sea (500 pictures)
- landscapes beach (500 pictures)
- landscapes island (500 pictures)
- landscapes japan (900 pictures)

Preprocessing: 
- Images were resized to 512×512.

Augmentations: 
- Random horizontal & vertical flips
- Random 15 degrees rotation
- Added color jitter
- Random resize crop

The dataset is not grouped into classes and only contains images so there's not much EDA or preprocessing to do apart from resizing the image to 512x512

## **Installation & Usage**

### Prerequisites
- Python 3.8+ (We used Python 3.13.9)

### Setup
- Clone the repo:

```git clone https://github.com/bavarianprayoga/landscapecolorizer```

- Install dependencies:

```pip install -r requirements.txt```

### Running the App 
Make sure you're on the root folder (not inside landscapecolorizer/app) then run:

```streamlit run app/app.py```

## **Limitations & Known Issues**

### Context Dependency:
The model has a strong dependency on global context cues, specifically the presence of a sky. When an image contains a clear sky (blue top / green bottom), colorization is vibrant and accurate. In zoomed-in images or scenes without a visible sky (e.g., dense forest close-ups), the model often reverts to desaturated "sepia" tones. This suggests the model relies on the sky as a positional anchor to trigger its "landscape" color palette.

![Output example](/outputs/plots/example_comparison_weakness.jpeg)
