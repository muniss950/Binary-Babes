import cv2
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

image_x, image_y = 200, 200

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    print(f"Folder '{folder_name}' created successfully.")

def main(c_id):
    starting_pic_no = 1000  # Starting number for pictures
    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    
    cap = cv2.VideoCapture(0)  # Changed camera index to 0
    create_folder("chords/" + str(c_id))
    pic_no = starting_pic_no

    while pic_no < starting_pic_no + 1000:  # Stop capturing after reaching 2000 images
        ret, image = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        image = cv2.flip(image, 1)
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
        res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            pic_no += 1
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            save_img = gray[y1:y1 + h1, x1:x1 + w1]
            save_img = cv2.resize(save_img, (image_x, image_y))
            cv2.putText(image, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
            save_path = os.path.join("chords", str(c_id), f"{pic_no}.jpg")
            cv2.imwrite(save_path, save_img)
            print(f"Image saved to: {save_path}")

            cv2.putText(image, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                break

        cv2.imshow("Capturing gesture", image)
        cv2.imshow("Res", res)

    cap.release()
    cv2.destroyAllWindows()

c_id = input("Enter chord: ")
main(c_id)
