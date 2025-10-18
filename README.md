# Assignment 3: User Interface for ML Models
## Absenteeism Prediction System

**Team Members:**
- Krishna Kumar Bais (241110038)
- Rohan (241110057)

**Course:** CS698Y - Human-AI Interaction

## Overview

This project extends Assignment 2 by providing a React-based user interface for the absenteeism prediction model. The interface enables HR professionals to make informed decisions while ensuring transparency, fairness, and responsible AI use.


## How to Run

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm

### 1) Backend setup
```bash
cd "Assignment 4"
python3 -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Start the Flask API
```bash
python app.py
```

### 3) Frontend
Open a new terminal:
```bash
cd "Assignment 4/frontend"
npm install
npm run build
npm run dev
```


## Project Structure

```
Assignment 4/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── main.jsx         # React entry point
│   ├── index.html           # HTML template
│   └── package.json         # Node.js dependencies
├── app.py                   # Flask backend
├── train_model.py           # Model training script
├── model.pkl                # Trained model (generated)
├── requirements.txt         # Python dependencies
├── Report.tex               # LaTeX report source
├── dev.sh                   # Development script
├── deploy.sh                # Production deployment
└── README.md                # This file
```

## Model Performance

- **RMSE**: 11.43 hours
- **MAE**: 6.44 hours
- **R² Score**: -0.20

### Fairness Metrics
- **Age Group MAE Gap**: 20.78 hours (Poor)
- **Education MAE Gap**: 13.36 hours (Moderate)
- **Service Time MAE Gap**: 3.76 hours (Good)
