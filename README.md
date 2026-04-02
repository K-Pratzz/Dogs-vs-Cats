🐶 Cats vs Dogs Classification (ML + Feature Engineering Approach)
📌 Overview
This project focuses on building an image classification model to distinguish between cats and dogs using traditional machine learning techniques.
Instead of directly using deep learning, the goal was to explore how far classical ML models can be pushed using feature engineering and optimization.
⚙️ Workflow
1. Data Preprocessing
Loaded images using OpenCV
Resized images to a fixed dimension
Normalized pixel values
Ensured consistent preprocessing between training and deployment
2. Feature Engineering (Key Improvement 🚀)
Initially:
Used flattened pixel values
Result: ~0.68 accuracy ❌
Improvement:
Implemented HOG (Histogram of Oriented Gradients)
Converted images to grayscale for better edge detection
Extracted meaningful structural features
👉 Result: Significant performance boost
3. Models Used
Support Vector Machine (SVM)
XGBoost
LightGBM
4. Hyperparameter Tuning
Used Optuna / GridSearchCV
Optimized model performance systematically
5. Ensemble Learning
Tested:
Voting Classifier
Stacking
Final choice:
Voting (XGBoost + LightGBM)
✔ Best balance of performance and simplicity
📊 Results
Model
Accuracy
SVM
0.73
XGBoost
0.74
LightGBM
0.74
Voting
0.75
Stacking
0.751
🚀 Key Learning
Raw pixels are weak features for ML models
Feature engineering (HOG) drastically improves performance
Ensemble methods provide marginal but useful gains
Classical ML has limitations on image data compared to CNNs
⚠️ Limitations
Does not capture spatial relationships like deep learning models
Performance plateau around ~75% accuracy
🧠 Future Improvements
Implement CNN / Transfer Learning
Compare ML vs Deep Learning performance
Improve feature extraction further
🛠️ Tech Stack
Python
NumPy, Pandas
OpenCV
Scikit-learn
XGBoost, LightGBM
Streamlit (for deployment)
