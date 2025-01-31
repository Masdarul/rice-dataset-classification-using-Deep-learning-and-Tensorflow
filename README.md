# **Rice Image Classification Using TensorFlow Model**
I used TensorFlow for classification using the [Rice Image Dataset by Murat KOKLU](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset), obtained from Kaggle. This dataset contains over 75,000 rice images with the same resolution. However, I adjusted the image resolution to ensure more accurate classification results. The trained model achieved 97% accuracy and 99% validation accuracy. I also tested the model by inputting images that match the corresponding labels or classes.
## **Process and Discussion**
There is a folder containing rice images with a resolution of 250x250. A Python program will read each image in this folder and then randomly resize it, ensuring that both the width and height are not less than 100 and not more than 500. Each image will have a different new resolution. The resized images will be saved in a separate folder that has been prepared beforehand. The resizing process is performed using a **while** loop.

| No | Image Name | Previous Resolution | New Resolution |
|----|------------|--------------------|-----------------|
| 1  | Gambar Arborio (1).jpg | 250x250 | 385x295 |
| 2  | Gambar Arborio (10).jpg | 250x250 | 190x391 |
| 3  | Gambar Arborio (100).jpg | 250x250 | 347x419 |
| 4  | Gambar Arborio (1001).jpg | 250x250 | 419x209 |
| 5  | Gambar Karacadag (9997).jpg | 250x250 | 201x497 |

After successfully resizing the rice images, the data will be split with a ratio of 80:20. The target folder containing the resized images will be moved to an empty folder that has been prepared beforehand. The splitting process will be done using the **splitfolders** library, where 80% of the data will go into **the training (train)** folder and 20% into **the validation (val)** folder.

After the **Image Data Generator** process, the **flow_from_directory** method is used to read images from the target folder. Images will be resized to 150x150, with a batch size of 64, and assigned labels based on their respective categories. Next, a **Sequential** model is built with the following layers:

| Layer Name     | Layer Details | Description/Function |
|---------------|--------------|---------------------|
| **Conv2D**   | `16, (3,3), padding='Same', activation='relu', input_shape=(150, 150, 3)` | Convolutional layer with 16 filters of size (3,3). 'Same' padding maintains the output size equal to the input. **ReLU** activation handles non-linearity. `input_shape=(150,150,3)` indicates an image size of 150x150 with 3 color channels (RGB). |
| **MaxPooling2D** | `(2,2)` | Pooling layer that reduces feature map size by taking the maximum value in each (2,2) kernel, decreasing image dimensions and improving computational efficiency. |
| **Conv2D**   | `32, (3,3), padding='Same', activation='relu'` | Second convolutional layer with 32 filters to extract more complex features. |
| **MaxPooling2D** | `(2,2)` | Pooling layer to further reduce feature map size. |
| **Conv2D**   | `64, (3,3), padding='Same', activation='relu'` | Third convolutional layer with 64 filters to capture more abstract features. |
| **MaxPooling2D** | `(2,2)` | Pooling layer to reduce feature map dimensions further. |
| **Flatten**  | - | Converts feature maps into a 1D vector for fully connected layers. |
| **Dense**    | `64, activation='relu'` | Fully connected layer with 64 neurons and **ReLU** activation to build complex relationships within the data. |
| **Dropout**  | `0.2` | Prevents overfitting by randomly dropping 20% of neurons during training. |
| **Dense**    | `5, activation='softmax'` | Output layer with 5 neurons for classification. **Softmax** activation is used to generate prediction probabilities. |

Once the model is created, it uses the **Adam optimizer**, **categorical_crossentropy loss**, and **accuracy metric**. The model is summarized using **model.summary(),** producing the following table:
| Layer (type)                 | Output Shape         | Param #      |
|-----------------------------|---------------------|--------------|
| conv2d (Conv2D)             | (None, 150, 150, 16) | 448          |
| max_pooling2d (MaxPooling2D) | (None, 75, 75, 16)   | 0            |
| conv2d_1 (Conv2D)           | (None, 75, 75, 32)   | 4,640        |
| max_pooling2d_1 (MaxPooling2D) | (None, 37, 37, 32) | 0            |
| conv2d_2 (Conv2D)           | (None, 37, 37, 64)   | 18,496       |
| max_pooling2d_2 (MaxPooling2D) | (None, 18, 18, 64) | 0            |
| flatten (Flatten)           | (None, 20736)       | 0            |
| dense (Dense)               | (None, 64)          | 1,327,168    |
| dropout (Dropout)           | (None, 64)          | 0            |
| dense_1 (Dense)             | (None, 5)           | 325          |

The model is trained using **``model.fit``** with **``train_generator``** for a maximum of 30 epochs. Validation is performed using **``val_generator``**, and EarlyStopping is applied as a callback. Training will stop early if both accuracy and validation accuracy exceed 96%, as the target has been achieved. In the first epoch, the model reached 90% accuracy, 25% loss, 97% validation accuracy, and 8% validation loss. In the second (final) epoch, accuracy improved to 97%, loss decreased to 7%, validation accuracy reached 99%, and validation loss dropped to 3%. The process stopped at epoch 2 (instead of 30) because the EarlyStopping callback detected that the stopping criteria (accuracy & validation accuracy exceeding 96%) were met. If the target had not been reached, training would have continued for up to **30 epochs**.

The final process involves converting the model to **TensorFlow Lite (TFLite), TensorFlow.js,** and **SavedModel**.

**``TensorFlow Lite (TFLite)``** is an open-source deep learning framework designed for on-device inference, making it ideal for deploying the YOLOv11 model on mobile, embedded, and IoT devices.

**``TensorFlow.js``** is an open-source hardware-accelerated JavaScript library used for training and deploying machine learning models directly in the browser or Node.js.
## **References for Further Experimentation and Research**
### **Code Experiments from Kaggle with the Same Dataset**
* **[Rice_Classification](https://www.kaggle.com/code/sam51700007/rice-classification)** 
*   **[Rice Classification CNN Model With 99.5% Accuracy](https://www.kaggle.com/code/youssefhelmy/rice-classification-cnn-model-with-99-5-accuracy)**
*   **[RiceTaskUAS](https://www.kaggle.com/code/riskiwulandari/berastugasuas)**
*   **[Rice Image Classification| PyTorch](https://www.kaggle.com/code/fatemehreihani/rice-image-classification-pytorch)**
*   **[Rice Image Classification Using Tensorflow](https://www.kaggle.com/code/fatemehreihani/rice-image-classification-using-tensorflow)**
*   **[Rice Classification with CNN in Keras](https://www.kaggle.com/code/mianbilal12/rice-classification-with-cnn-in-keras)**
*   **[LeNet5 - Classification On rice-image-dataset](https://www.kaggle.com/code/nalinp/lenet5-classification-on-rice-image-dataset)**
*   **[VGG16 model with SVM Classifier | 97.2% Accuracy](https://www.kaggle.com/code/shubanms/vgg16-model-with-svm-classifier-97-2-accuracy)**
### **Official Documentation References**
*   **[TensorFlow](https://www.tensorflow.org/learn)**
*   **[Keras](https://keras.io/guides/)**
*   **[Adam](https://keras.io/api/optimizers/adam/)**
*   **[ReLU](https://www.tensorflow.org/api_docs/python/tf/nn/relu)**
*   **[TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite)**
*   **[TensorFlow JavaScript](https://www.tensorflow.org/js)**
### **Additional Learning Resources**
*   **[Deep Learning By Coursera And DeepLearning.AI](https://www.coursera.org/specializations/deep-learning)**
*   **[TensorFlow Developer By Coursera And DeepLearning.AI](https://www.coursera.org/professional-certificates/tensorflow-in-practice)**
*   **[TensorFlow: Advanced Techniques By Coursera And DeepLearning.AI](https://www.coursera.org/specializations/tensorflow-advanced-techniques)**
*   **[Learn Machine Learning Development By Dicoding](https://www.dicoding.com/academies/185-belajar-pengembangan-machine-learning)**
*   **[Applied Machine Learning By Dicoding](https://www.dicoding.com/academies/319-machine-learning-terapan)**
*   **[Deep Learning By Edx And IBM](https://www.edx.org/certificates/professional-certificate/ibm-deep-learning?index=product&queryId=50c4a7b627a3fe40f17d71e9cdb9a46e&position=3)**
