# Cyber-Threat-Detection

A machine learning-based Cyber Threat Detection System that analyzes network traffic and classifies it as **benign or malicious** using multiple machine learning models.

## 🚀 Project Overview

Cyber-Threat-Detection is an intelligent cybersecurity application developed using **Python, Machine Learning, and Flask**.

The system analyzes network traffic data, performs preprocessing and feature analysis, and uses trained machine learning models to identify potentially malicious network activity.

The project combines multiple machine learning techniques to improve the accuracy and reliability of cyber threat detection.

## ✨ Features

* 🔐 Detects potentially malicious network traffic
* 📊 Analyzes network traffic data
* 🤖 Uses multiple machine learning models
* 🧠 Includes anomaly detection
* ⚙️ Performs data preprocessing and feature engineering
* 🌐 Provides a Flask-based web interface
* 📈 Supports intelligent classification of network traffic

## 🛠️ Technologies Used

* **Python**
* **Machine Learning**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Flask**
* **HTML**
* **CSS**
* **JavaScript**

## 🤖 Machine Learning Models

The project includes trained models such as:

* Random Forest
* Extra Trees
* Gradient Boosting
* Voting Classifier
* Stacking Classifier
* Anomaly Detection Model

The trained models are stored as `.pkl` files and are used by the application for prediction.

## 📂 Project Structure

```text
Cyber-Threat-Detection/
│
├── static/
├── templates/
│
├── analysis.py
├── app.py
├── train_model.py
│
├── anomaly_detector.pkl
├── scaler.pkl
├── stacking_model.pkl
├── voting_model.pkl
│
├── network_traffic_data.csv
├── requirement.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Cyber-Threat-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirement.txt
```

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open the application in your browser:

```text
http://127.0.0.1:5000
```

## 🔍 How It Works

The system follows a machine learning-based detection pipeline:

```text
Network Traffic Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Feature Scaling
        ↓
Machine Learning Models
        ↓
Threat Classification
        ↓
Benign / Malicious
```

## 🎯 Objective

The main objective of this project is to develop an intelligent system capable of identifying suspicious or malicious network traffic using machine learning techniques.

## 🔮 Future Improvements

* Real-time network traffic monitoring
* Integration with live packet capture
* Improved threat classification
* Real-time alert notifications
* Advanced visualization and analytics
* Deployment on cloud infrastructure

## 👩‍💻 Author

**Sneha Bishnoi**

B.Tech Computer Engineering

## 📄 License

This project is developed for educational and academic purposes.

