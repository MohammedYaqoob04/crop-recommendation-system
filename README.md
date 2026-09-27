# 🌱 Crop Recommendation System

An AI-based crop recommendation system that recommends a suitable crop based on soil nutrients and environmental conditions using Machine Learning.

## 📌 Project Overview

The Crop Recommendation System uses a Random Forest classification model to recommend a suitable crop based on seven agricultural parameters:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

The system is trained using the Crop Recommendation dataset containing **2,200 samples across 22 crop categories**.

## 🚀 Features

* 🌱 Crop recommendation using Machine Learning
* 🌳 Random Forest classification
* 📊 Exploratory Data Analysis (EDA)
* 📈 Feature importance visualization
* 🥇 Top 3 crop recommendations
* ⚡ Interactive Streamlit web application
* ✅ Input validation
* 💾 Saved trained model using Joblib

## 🧠 Machine Learning

Several classification algorithms were experimented with during development:

| Model         |   Accuracy |
| ------------- | ---------: |
| Decision Tree |     97.95% |
| KNN           |     97.73% |
| Random Forest | **99.55%** |

Random Forest was selected as the final model based on the test-set accuracy obtained during experimentation.

### Dataset

The dataset contains:

* **2,200 samples**
* **7 input features**
* **22 crop classes**
* No missing values
* No duplicate rows
* 100 samples per crop class

The dataset is balanced, with an equal number of samples for each crop category.

## 📊 Model Feature Importance

The trained Random Forest model identified the following overall feature importance:

| Feature        | Importance |
| -------------- | ---------: |
| Rainfall       |     23.02% |
| Humidity       |     22.42% |
| Potassium (K)  |     17.54% |
| Phosphorus (P) |     15.09% |
| Nitrogen (N)   |      9.64% |
| Temperature    |      7.24% |
| Soil pH        |      5.06% |

These values represent **model-specific feature importance** and should not be interpreted as direct biological causation.

## 🏗️ Project Structure

```text
crop-recommendation-system/
│
├── dataset/
│   └── crop_data.csv
│
├── model/
│   └── crop_model.pkl
│
├── notebooks/
│   └── crop_eda.ipynb
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd crop-recommendation-system
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Make sure the trained model exists at:

```text
model/crop_model.pkl
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## 🔄 Retraining the Model

To retrain the Random Forest model using the dataset:

```bash
python train_model.py
```

The newly trained model will be saved to:

```text
model/crop_model.pkl
```

## ⚠️ Limitations

The model was trained and evaluated on a clean, balanced dataset. Therefore, the reported test accuracy may not represent performance under real-world agricultural conditions.

The dataset's balanced and relatively well-separated classes may result in higher test accuracy than would be obtained with noisier real-world soil and environmental measurements.

The recommendations should therefore be considered **machine-learning-based suggestions rather than guaranteed crop-success predictions**.

The probability values displayed by the application are Random Forest model probability estimates and should not be interpreted as guaranteed probabilities of successful crop cultivation.

## 🔮 Future Enhancements

* Real-time soil sensor integration
* ESP32-based data collection
* Weather API integration
* Crop yield prediction
* Fertilizer recommendation
* Disease detection using computer vision
* Cloud-based monitoring
* Mobile application

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Machine Learning

## 📜 License

This project is developed for educational and academic purposes.

## 👨‍💻 Author

**Mohammed Yaqoob S**
Artificial Intelligence and Data Science Student
