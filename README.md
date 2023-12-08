# Image_Classifier
# Image Classification Model
## 1. Model Overview
The image classification model is designed to recognize and classify images of
celebrities using Convolutional Neural Networks (CNNs). The dataset comprises
images of five sports celebrities: Lionel Messi, Maria Sharapova, Roger Federer,
Serena Williams, and Virat Kohli.
## 2. Data Preprocessing
• Dataset: Images were collected for each celebrity from the provided directory.
• Preprocessing: Images were resized to (128x128) pixels and converted to
NumPy arrays to be fed into the CNN model.
## 3. Model Architecture
The CNN model architecture consists of the following layers:
• Convolutional Layers: Three sets of Conv2D and MaxPooling2D layers to
extract features from the images.
• Dropout Layer: A dropout layer with a dropout rate of 0.2 for regularization.
• Flatten Layer: Flattening the output to be fed into Dense layers.
• Dense Layers: Two dense hidden layers with ReLU activation functions and a
final output layer with Softmax activation for multi-class classification.
## 4. Training Process
• Train-Test Split: The dataset was divided into training and testing sets with a
80:20 split.
• Normalization: Image pixel values were normalized to the range [0, 1].
• Model Compilation: Adam optimizer was used with Sparse Categorical
Crossentropy loss function.
• Training: The model was trained for 25 epochs with a batch size of 64, using a
validation split of 20%.
• Training Metrics: Accuracy and loss were monitored during training to assess
model performance.
## 5. Model Evaluation
• Accuracy: After training, the model achieved an accuracy of 73.35% on the test
dataset.
## 6. Model Prediction
• Prediction: A prediction function was implemented to predict the celebrity
from a given image path using the trained model.
## 7. Results and Findings
• Training Plots: Plots were generated for training accuracy vs. validation
accuracy and training loss vs. validation loss.
• CSV Output: Model predictions were saved to a CSV file ('Image_CNN.csv')
containing the actual and predicted labels for the test dataset.
## Conclusion
The image classification model successfully identifies and classifies images of
celebrities with an accuracy of 73.53%. Further enhancements or fine-tuning could be
considered for improved performance.
