import cv2
import pickle
import cvzone 
import numpy as np
from dynamic_pricing import DynamicPricingModel, create_training_data
import datetime

# Initialize Dynamic Pricing Model
pricing_model = DynamicPricingModel()
training_data = create_training_data()
X = training_data[['distance', 'traffic_level', 'hour', 'is_weekend', 'available_slots']].values
y = training_data['price'].values
pricing_model.train(X, y)

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def calculate_current_price(available_spaces, total_spaces):
    """Calculate dynamic price based on current conditions"""
    current_hour = datetime.datetime.now().hour
    is_weekend = 1 if datetime.datetime.now().weekday() >= 5 else 0
    
    # Example features: distance from center (5km), current traffic (70%), 
    # current hour, is_weekend, available spaces percentage
    features = np.array([
        5.0,  # distance from city center (km)
        70.0,  # traffic level (%)
        float(current_hour),
        float(is_weekend),
        float(available_spaces)
    ])
    
    return pricing_model.predict_price(features)

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Calculate current price
    current_price = calculate_current_price(spaceCounter, len(posList))
    
    # Display available spaces and current price
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))
    cvzone.putTextRect(img, f'Price: ₹{current_price:.2f}/hr', (100, 100), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgMedian)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()