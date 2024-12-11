# Segmentation and Detection using YOLO and U-Net

This project implements both **semantic segmentation** and **object detection** using two powerful deep learning models: **YOLO** (You Only Look Once) for object detection and **U-Net** for semantic segmentation. It uses the **Oxford-IIIT Pet Dataset**, which contains pixel-wise annotated images of 37 different pet categories.

## 📝 What It Is

This project uses two neural networks:
1. **YOLOv3** for object detection: Detects and locates pets in images, outputting bounding boxes and class labels.
2. **U-Net** for semantic segmentation: Provides pixel-wise segmentation of pets in the images.

The model works on a dataset of images and segmentation masks, where each pixel in the segmentation mask corresponds to a part of an object (i.e., a pet) in the image.

### 🛠 How It’s Made

#### **Data Preprocessing**
The **Oxford-IIIT Pet Dataset** is used, which consists of images and their corresponding segmentation masks. The images are resized to 128x128 for training the models. The segmentation masks are processed to convert multi-class labels into binary masks for simplicity.

The preprocessing steps involve:
1. **Resizing** images and segmentation masks to a uniform size (128x128).
2. **Normalization** of pixel values to a [0, 1] range.
3. **Binary mask transformation** for segmentation (if needed).

#### **U-Net Model (Segmentation)**
The **U-Net** architecture consists of an encoder (downsampling) and a decoder (upsampling). The network learns to perform pixel-wise classification using a series of convolutional layers. The architecture also includes skip connections, which help preserve spatial features lost during downsampling.

#### **YOLO Model (Detection)**
The **YOLOv3** model performs real-time object detection. The input image is processed by the network to predict bounding boxes for the detected objects, followed by non-max suppression to eliminate redundant boxes. The model is trained to predict class probabilities and the bounding box coordinates.

#### **Model Training**
The models are trained using the processed datasets for **10 epochs** with the following configurations:
- **Optimizer**: Adam
- **Loss Function**: 
  - For U-Net: Combined loss with **Dice Coefficient** and **Binary Cross-Entropy**.
  - For YOLO: Custom loss based on bounding box prediction and classification accuracy.
- **Metrics**: Accuracy and Dice Coefficient for segmentation and detection tasks.

#### **Results**
- **Training Accuracy** and **Loss** are monitored using TensorFlow's `fit()` method.
- The **mAP (mean Average Precision)** is calculated at multiple IoU (Intersection over Union) thresholds to evaluate the performance of the YOLO model on the test set.

### 🖼 Example Results

#### **Segmentation Output**
For the segmentation model, an image with the predicted mask is compared with the ground truth mask:

```python
Epoch 1/20
168/168 ━━━━━━━━━━━━━━━━━━━━ 116s 461ms/step - dice_coefficient: 0.4604 - loss: 1.5286 - val_dice_coefficient: 0.5824 - val_loss: 1.0219
