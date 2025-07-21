# Glaucoma Classification using MobileNetV2, Grad-CAM, and Segmentation

This deep learning project performs **glaucoma classification** from eye fundus images using a pre-trained MobileNetV2 model. It also includes **Grad-CAM visualization** for model explainability and **OpenCV-based optic disc segmentation** to highlight affected regions.

## Features
- Eye image classification into glaucoma or normal
- Transfer learning using MobileNetV2
- Grad-CAM for visual explainability
- Optic disc/tumor region segmentation (basic thresholding)
- AUC-ROC, Precision, Recall, Confusion Matrix for evaluation

## Dataset Structure
Place your dataset in the following structure:
/content/drive/MyDrive/dataset/
├── glaucoma/
│ ├── g1.jpg
│ ├── g2.jpg
├── normal/
│ ├── n1.jpg
│ ├── n2.jpg


## Model
The model uses MobileNetV2 (pre-trained on ImageNet) with a custom classification head:
- GlobalAveragePooling
- Dense Layer + Dropout
- Softmax output

## How to Run

### 1. Install dependencies:
pip install -r requirements.txt

##Train the Model
model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[...])

##Predict the values
predict_and_segment("/path/to/test_image.jpg")


