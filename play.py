from keras.models import load_model
import cv2
import numpy as np
import pyautogui

REV_CLASS_MAP = {
    0:"down",
    1:"left",
    2:"right",
    3:"up",
    4:"none"
    }

def mapper(val):
    return REV_CLASS_MAP[val]

def play(gesture):
    if gesture == "left":
        pyautogui.press('left') 
        # pyautogui.hotkey('ctrl','left')  # for vlc player

    elif gesture == "up":
        # pyautogui.press('up') 
        pyautogui.press('0')
        # pyautogui.press('volumeup')

        # pyautogui.hotkey('fn','f3')
     
    elif gesture == "right":
        pyautogui.press('right') 
        # pyautogui.hotkey('ctrl','right')
        
    
    elif gesture == "down":
        # pyautogui.press('down') 
        pyautogui.press('9')
        # pyautogui.press('volumedown')



kernel = np.ones((5,5),np.uint8)

model = load_model("hand_gesture.h5")

cap = cv2.VideoCapture(0)

prev_move = None

FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)

    cv2.rectangle(frame, (300, 50), (600, 350), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[50:350, 300:600]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    gesture_code = np.argmax(pred[0])
    user_gesture = mapper(gesture_code)
    # play(user_gesture)
    
    if user_gesture == "left":
        cv2.putText(frame, "Reverse..",(5, 50), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        pyautogui.press('left') 

    elif user_gesture == "up":
        cv2.putText(frame, "Volume Up..",(5, 50), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        pyautogui.press('0')
     
    elif user_gesture == "right":
        cv2.putText(frame, "Forward..",(5, 50), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        pyautogui.press('right') 
        
    
    elif user_gesture == "down":
        cv2.putText(frame, "Volume Down..",(5, 50), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        pyautogui.press('9')

    cv2.imshow("frame", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
