from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
from dynamic_pricing import DynamicPricingModel, create_training_data
import datetime
import base64
import pickle
import cvzone
import os

app = Flask(__name__)

# Initialize Dynamic Pricing Model
pricing_model = DynamicPricingModel()
training_data = create_training_data()
X = training_data[['distance', 'traffic_level', 'hour', 'is_weekend', 'available_slots']].values
y = training_data['price'].values
pricing_model.train(X, y)

# Load parking space positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

def calculate_current_price(available_spaces, total_spaces):
    """Calculate dynamic price based on current conditions"""
    current_hour = datetime.datetime.now().hour
    is_weekend = 1 if datetime.datetime.now().weekday() >= 5 else 0
    
    features = np.array([
        5.0,  # distance from city center (km)
        70.0,  # traffic level (%)
        float(current_hour),
        float(is_weekend),
        float(available_spaces)
    ])
    
    return pricing_model.predict_price(features)

def process_frame(frame):
    """Process a video frame and return parking space info"""
    if frame is None:
        return None, 0
        
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    spaceCounter = 0
    width, height = 107, 48

    for pos in posList:
        x, y = pos
        imgCrop = imgDilate[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(frame, str(count), (x, y + height - 3), scale=1,
                       thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(frame, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                   thickness=5, offset=20, colorR=(0,200,0))
    
    current_price = calculate_current_price(spaceCounter, len(posList))
    cvzone.putTextRect(frame, f'Price: ₹{current_price:.2f}/hr', (100, 100), scale=3,
                   thickness=5, offset=20, colorR=(0,200,0))

    return frame, spaceCounter

def gen_frames():
    """Video streaming generator function."""
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carPark.mp4')
    cap = cv2.VideoCapture(video_path)
    
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame, _ = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_parking_status')
def get_parking_status():
    """Get current parking status and price"""
    # Use absolute path for video file
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carPark.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({'error': f'Could not open video file at {video_path}'})
            
        # Check if we're at the end of the video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        success, frame = cap.read()
        if not success:
            return jsonify({'error': 'Failed to read video frame'})

        processed_frame, available_spaces = process_frame(frame)
        total_spaces = len(posList)
        current_price = calculate_current_price(available_spaces, total_spaces)

        # Convert frame to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'frame': frame_b64,
            'available_spaces': available_spaces,
            'total_spaces': total_spaces,
            'current_price': round(current_price, 2),
            'is_weekend': datetime.datetime.now().weekday() >= 5,
            'current_hour': datetime.datetime.now().hour
        })
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'})
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == '__main__':
    app.run(debug=True)
