# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mediapipe",
#     "opencv-python",
#     "cvzone",
#     "google-genai",
# ]
# ///

from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from google import genai
from PIL import Image


client = genai.Client(api_key="AIzaSyDCsdlR9WU6JM9MDMh9ZnvcAGnbhmJDRbQ")
# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHand(img):
     
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        # print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up
        print(fingers1)
        return fingers1,lmList1
    else:
        return None

def draw(info,prev,canvas):
    fingers,lmlist = info
    curr = None
    if fingers == [0,1,0,0,0]:
        curr = lmList[8][0:2]
        if prev is None: prev = curr
        cv2.line(canvas,curr,prev,(255,0,255),10)
    elif fingers == [0, 0, 0, 0, 1]:
        canvas[:] = 0  # Reset canvas to black
        prev = None
    return curr
# Continuously get frames from the webcam
def sendtoAI(canvas,fingers):
    if fingers == [1,1,1,0,0]:
        image = Image.fromarray(canvas)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, "Solve the math problem"]
        )
        print(response.text)

prev = None
canvas = None
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img,1)
    if canvas is None:
        canvas = np.zeros_like(img)
    info = getHand(img)
    if info:
        fingers,lmList = info
        print(fingers)
        prev = draw(info,prev,canvas)
        sendtoAI(canvas,fingers)
    img_com = cv2.addWeighted(img,0.7,canvas,0.3,0)
    # Check if any hands are detected
    # Display the image in a window
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Combined", img_com)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)