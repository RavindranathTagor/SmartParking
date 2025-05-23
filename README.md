# Smart Parking System

This project is a Smart Parking System web application that provides a live parking view, dynamic pricing, and real-time availability updates. It uses Python, Flask, OpenCV, and Bootstrap for the frontend.

## Features
- Live video feed of parking area
- Real-time parking space availability
- Dynamic pricing based on time, day, and occupancy
- Modern, responsive UI with Bootstrap

## Prerequisites
- Python 3.10+
- Git
- (Recommended) Virtual environment tool (venv)

## Setup Instructions

### 1. Clone the Repository
```
git clone <repo-url>
cd CarParkProject
```

### 2. Create and Activate Virtual Environment (Optional but recommended)
```
python -m venv newenv
# On Windows:
newenv\Scripts\activate
# On Linux/Mac:
source newenv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```
pip install flask opencv-python cvzone numpy pandas matplotlib
```

### 4. Run the Application
```
python app.py
```

The app will start on `http://127.0.0.1:5000/` by default.

## Project Structure
- `app.py` : Main Flask application
- `templates/index.html` : Frontend HTML
- `CarParkPos/` : Parking position data
- `carPark.mp4`, `carParkImg.png` : Video/image resources
- `dynamic_pricing.py` : Dynamic pricing logic
- `parking_data.csv` : Parking data

