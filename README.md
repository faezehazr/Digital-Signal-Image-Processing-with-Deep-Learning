# 🌟 Digital Signal & Image Processing with Deep Learning 🚀  

📊 **Exploring Deep Learning for Signal Classification, Image Retrieval, and GANs!**  

---

## 🔍 Project Overview  
This project applies **deep learning and computer vision** techniques to **ECG classification, image retrieval, and generative modeling**. The goal is to develop models that analyze **ECG signals, classify flower images, and generate synthetic images using GANs**.  

📌 **Key Highlights:**  
- 🏥 **ECG Arrhythmia Classification** – CNN-LSTM hybrid model for detecting abnormal heartbeats.  
- 🌸 **Flower Classification** – CNN and transfer learning to classify **102 flower species**.  
- 🔎 **Content-Based Image Retrieval (CBIR)** – Using deep features for **image similarity search**.  
- 🎨 **Generative Adversarial Networks (GANs)** – Training a **DCGAN** to generate synthetic flower images.  

---

## 📂 Dataset Sources  
📌 **MIT-BIH Arrhythmia Dataset** – ECG data for arrhythmia classification.  
📌 **102 Flowers Dataset** – Used for training CNN-based image classifiers.  
📌 **Custom Image Dataset** – Used for GAN-based image generation.  

🔗 Public datasets available on **Kaggle** and **official repositories**.  

---

## 🛠 Technologies Used  
✅ **Python** 🐍  
✅ **TensorFlow & Keras** – Deep learning frameworks  
✅ **OpenCV** – Image processing  
✅ **Scikit-learn** – Machine learning utilities  
✅ **NumPy & Pandas** – Data handling  
✅ **Matplotlib & Seaborn** – Data visualization  
✅ **PyWavelets** – ECG signal preprocessing  

---

## 🚀 How to Run the Project  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/digital-signal-image-processing.git
cd digital-signal-image-processing
## 📊 Model Performance & Evaluation  
The models were evaluated using **accuracy, confusion matrices, and precision-recall metrics**.  

### 📈 Results:  

| Model                   | Accuracy  | Precision | Recall  | F1 Score |
|-------------------------|----------|----------|--------|----------|
| **CNN-LSTM (ECG)**      | **98.69%** | **99%**  | **98%** | **98.5%** |
| **CNN (Flowers)**       | **47.3%** | **46%**  | **48%** | **47%** |
| **CBIR (InceptionV3)**  | **0.95 Precision** | **0.046 F1-score** | **Fastest Execution (~152s)** |  |
| **GAN (Generated Images)** | *Realistic floral-like patterns (needs further fine-tuning)* |  |  |  |

✅ **CNN-LSTM outperformed other ECG models.**  
✅ **InceptionV3 was the best model for image retrieval.**  
✅ **GANs successfully generated images, but further training is needed for clarity.**  
