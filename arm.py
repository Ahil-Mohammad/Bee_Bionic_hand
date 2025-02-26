import cv2
import mediapipe as mp
import pyfirmata

# Set up Arduino board and servo pin
board = pyfirmata.Arduino('COM5')  # Replace 'COM3' with the port your Arduino is connected to
servo_pin = 2  # Replace with the actual pin connected to the servo
servo = board.get_pin('d:{}:s'.format(servo_pin))

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmList = []
        fingers = []

        if results.multi_hand_landmarks:
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mp_draw.draw_landmarks(image, myHands, mp_hand.HAND_CONNECTIONS)

            # Dynamically calculate tip indices based on the length of lmList
            tipIds = [4, 8, 12, 16, 20]
            tipIds = [tip_id for tip_id in tipIds if tip_id < len(lmList)]

            for id in range(len(tipIds)):
                # Modify the condition for the thumb
                if id == 0:  # Check if the finger is the thumb
                    if lmList[tipIds[id]][1] > lmList[tipIds[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            total = fingers.count(1)

            # Control the servo based on finger state
            if total == 5:
                # All fingers closed, rotate to 180 degrees
                servo.write(0)
            elif total == 0:
                # All fingers open, rotate to 0 degrees
                servo.write(180)

            # Display the state of each finger
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i in range(len(tipIds)):
                if len(fingers) > i:
                    finger_state = "Open" if fingers[i] == 1 else "Closed"
                    cv2.putText(image, f'{finger_names[i]}: {finger_state}', (20, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 0), 2)

        cv2.imshow("Arm recognizor", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            board.exit()
            break

video.release()
cv2.destroyAllWindows()