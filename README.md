# Parkinson's Disease Detection using Machine Learning

## Overview

This project focuses on detecting Parkinson's Disease using Machine Learning algorithms. Parkinson’s Disease is a progressive neurological disorder that affects movement and motor control. Early and accurate diagnosis is crucial to managing symptoms and improving quality of life. However, traditional diagnosis often relies heavily on expert evaluation and may not be easily accessible to everyone. With the help of machine learning, we can build models that assist medical professionals by providing reliable early-stage predictions.

For this project, I used a publicly available dataset from **Kaggle** and developed the machine learning models on **Google Colab**, taking advantage of its free GPU/CPU resources and collaborative features.  

---

## Dataset

- **Source**: Kaggle (Parkinson’s Disease Dataset)
- **Description**: The dataset consists of biomedical voice measurements from individuals with and without Parkinson’s Disease.
- **Features**: 
  - Total of 22 columns including:
    - **MDVP:Fo(Hz)** – Average vocal fundamental frequency
    - **MDVP:Jitter(%)**, **MDVP:Jitter(Abs)** – Measures of variation in frequency
    - **MDVP:Shimmer**, **MDVP:Shimmer(dB)** – Measures of variation in amplitude
    - **NHR, HNR** – Noise-to-harmonics and harmonics-to-noise ratios
    - **RPDE, DFA** – Nonlinear dynamical complexity measures
    - **spread1, spread2, D2** – Nonlinear measures of signal
    - **status** – Target variable (1 = Parkinson's, 0 = Healthy)
  - **Instances**: 195 samples
  
The data is already labeled, which makes it a **supervised classification** problem.

---

## Project Workflow

1. **Data Collection**  
   - Downloaded the Parkinson’s dataset from Kaggle.
   - Uploaded it to Google Drive for easy access in Google Colab.

2. **Environment Setup**  
   - Platform: **Google Colab** (because of its ease of use and free resources)
   - Language: **Python 3**
   - Key Libraries:
     - `pandas` for data manipulation
     - `numpy` for numerical operations
     - `sklearn` for machine learning models
     - `matplotlib`, `seaborn` for visualization

3. **Data Preprocessing**
   - Loaded the dataset into a Pandas DataFrame.
   - Checked for missing values and found none.
   - Performed feature scaling using **StandardScaler** to normalize the data because different features had varying ranges.
   - Separated the features (`X`) and target variable (`y`).

4. **Exploratory Data Analysis (EDA)**
   - Analyzed feature correlations with a heatmap.
   - Found that features like `spread1`, `DFA`, `PPE`, and `RPDE` had strong relationships with the target variable.
   - Visualized the distributions of key features for better understanding.

5. **Model Building**
   - Split the data into training and testing sets (80%-20%).
   - Implemented several classification models:
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**
     - **Random Forest Classifier**
     - **K-Nearest Neighbors (KNN)**
   - Compared models based on accuracy, precision, recall, and F1-score.

6. **Model Evaluation**
   - Used confusion matrices and classification reports.
   - Plotted ROC curves for visual comparison.
   - The **SVM model** gave the highest accuracy (around 94% in my case).
   - Random Forest also performed very well but slightly less than SVM.

7. **Conclusion**
   - Machine Learning models, especially SVM, can be highly effective for predicting Parkinson's Disease based on vocal measurements.
   - Even with relatively small datasets, careful preprocessing and model selection can result in very high prediction accuracy.

---

## Key Results

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 87%      |
| KNN                  | 90%      |
| Random Forest        | 92%      |
| SVM                  | 94%      |

- **Support Vector Machine (SVM)** emerged as the best-performing model.
- Using proper scaling and hyperparameter tuning could further improve these results.

---

## Challenges Faced

- **Small Dataset**: Working with just 195 samples means the model could potentially overfit. Used techniques like cross-validation to help mitigate this.
- **Imbalanced Features**: Some features had much higher variances, requiring careful normalization.
- **Choosing the Right Model**: Initially, KNN seemed good, but SVM provided better and more stable results.

---

## Future Work

- **Hyperparameter Tuning**: Use GridSearchCV to find the best parameters for SVM and Random Forest.
- **Feature Engineering**: Try new feature extraction techniques to improve performance.
- **Deep Learning**: Implement a simple neural network using TensorFlow or PyTorch and compare with traditional machine learning models.
- **Deployment**: Build a simple Flask or Streamlit web app where users can input their values and get predictions.
- **Larger Datasets**: Incorporate larger and more diverse datasets for more generalized models.

---

## How to Run

1. Open **Google Colab**.
2. Upload the dataset or mount your Google Drive.
3. Install necessary libraries (`pip install pandas numpy scikit-learn matplotlib seaborn` if needed).
4. Load the dataset.
5. Run through the cells for preprocessing, EDA, model building, and evaluation.
6. Try your own variations: tune parameters, add models, or visualize in new ways!

---

## Acknowledgments

- Dataset provided by Kaggle community contributors.
- Project heavily inspired by research articles on early Parkinson’s Disease detection.
- Thanks to Google Colab for providing free cloud resources!

