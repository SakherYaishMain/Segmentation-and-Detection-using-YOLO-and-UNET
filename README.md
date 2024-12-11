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
```
### Object Detection Output

For the object detection model, bounding boxes are drawn around detected pets with the associated class labels and confidence scores:
```plain
Testing Accuracy: 85.72%
```
The bounding boxes are drawn around detected objects, showing their confidence and the correct identification of pet breeds.

## 📊 Data

The dataset used consists of the following columns:
- **image**: The pet image.
- **segmentation_mask**: The pixel-wise ground truth mask for segmentation.
- **label**: The category label (breed of pet).
- **species**: The species of the animal (dog or cat).

The dataset is preprocessed by resizing images and segmentation masks, normalizing pixel values, and converting the segmentation masks to binary format for segmentation tasks.

## 🤖 Technologies Used

- **Python 3.x**: The primary programming language used for implementation.
- **TensorFlow**: For building and training deep learning models.
- **Keras**: For constructing U-Net and YOLO architectures.
- **NumPy**: For numerical operations and data manipulation.
- **Matplotlib**: For visualization of results and model predictions.
- **OpenCV**: For image preprocessing and handling.
- **scikit-learn**: For evaluation metrics (mean Average Precision).

## 📈 Model Evaluation

The models are evaluated using multiple metrics:

- **mAP (mean Average Precision)**: Used for evaluating object detection performance at different IoU thresholds (e.g., 0.25, 0.5, 0.75).
- **Dice Coefficient**: A metric used for evaluating the accuracy of pixel-wise segmentation.
- **Accuracy**: The overall performance of both the YOLO and U-Net models.

### Example mAP Scores:
```plain
mAP@0.25: 85.2% mAP@0.5: 79.1% mAP@0.75: 70.4% mAP@0.95: 65.6%
```

## ⚡ Model Results Visualization

### Training and Validation Loss

A plot is generated to visualize the loss curves for both training and validation sets.

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coefficient'], label='Training Accuracy')
plt.plot(history.history['val_dice_coefficient'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()
```
### Detection Bounding Boxes

After inference, the detected objects (pets) are shown with bounding boxes, confidence scores, and class labels.

## 🧑‍💻 Future Improvements

- **Model Optimization**: Explore other architectures like **DeepLab** or **Mask R-CNN** for improved segmentation results.
- **Data Augmentation**: Apply more robust data augmentation techniques to increase model generalization.
- **Real-time Detection**: Integrate the models into a real-time application for detecting and segmenting pets in live video feeds.
