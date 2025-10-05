Employee Performance Prediction

A simple Flask + Machine Learning web app that predicts employee performance.

🚀 Features

* Flask backend

* Pre-trained ML model (model.pkl)

* HTML templates + CSS styling

* Easy to run locally or deploy on Render

📂 Project Structure
EmployeePerformancePrediction/
│
├── app.py                # Flask app (entry point)
├── templates/            # HTML templates
│   ├── index.html
│   └── result.html
├── static/               # CSS, JS, images
│   └── style.css
├── model.pkl             # Trained ML model
├── requirements.txt      # Dependencies
├── Procfile              # For Render deployment
├── runtime.txt           # Python version
└── README.md             # This file

⚙️ Installation & Running Locally

* Clone the repository

git clone https://github.com/your-username/EmployeePerformancePrediction.git
cd EmployeePerformancePrediction


* Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux


* Install dependencies

pip install -r requirements.txt


* Run the Flask app

python app.py


*Open in browser
Go to 👉 http://127.0.0.1:5000
Screenshots
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/f95132e1-05d1-4447-afc8-b467709c2cdc" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/ebdd432f-deb8-4020-ade6-56b6025010f5" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/9aa79793-bd0b-4d57-8b15-57fe458445c7" />
