import cv2
import mediapipe as mp


def detectaMano():

    mp_manos = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    with mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        if x < x_min:
                            x_min = x
                        if x > x_max:
                            x_max = x
                        if y < y_min:
                            y_min = y
                        if y > y_max:
                            y_max = y
                    
                    expand_factor = 0.2  # Adjust this value to control the expansion
                    x_min = max(0, x_min - int((x_max - x_min) * expand_factor))
                    y_min = max(0, y_min - int((y_max - y_min) * expand_factor))
                    x_max = min(frame.shape[1], x_max + int((x_max - x_min) * expand_factor))
                    y_max = min(frame.shape[0], y_max + int((y_max - y_min) * expand_factor))
                    
                    # Draw rectangle around the bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_manos.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,0,255)),
                        mp_drawing.DrawingSpec(color=(0,255,0)))
                    

            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

detectaMano()