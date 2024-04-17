import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PoseStateMemento:
    def __init__(self, up, down, count):
        self.up = up
        self.down = down
        self.count = count

class PoseDetector:
    def __init__(self, video_path, new_width=700, new_height=800):
        self.video_path = video_path
        self.new_width = new_width
        self.new_height = new_height
        self.up = False
        self.down = False
        self.count = 0
        self.mp_pose = mp.solutions.pose

    def start_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        with self.mp_pose.Pose(static_image_mode=False) as pose:
            while True:
                ret, frame = cap.read()
                if ret == False:
                    break

                frame = cv2.resize(frame, (self.new_width, self.new_height))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(frame_rgb)

                if results.pose_landmarks is not None:
                    x1 = int(results.pose_landmarks.landmark[24].x * self.new_width)
                    y1 = int(results.pose_landmarks.landmark[24].y * self.new_height)

                    x2 = int(results.pose_landmarks.landmark[26].x * self.new_width)
                    y2 = int(results.pose_landmarks.landmark[26].y * self.new_height)

                    x3 = int(results.pose_landmarks.landmark[28].x * self.new_width)
                    y3 = int(results.pose_landmarks.landmark[28].y * self.new_height)

                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])

                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    if angle >= 160:
                        self.up = True
                    if self.up and not self.down and angle <= 70:
                        self.down = True
                    if self.up and self.down and angle >= 160:
                        self.count += 1
                        self.up = False
                        self.down = False

                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)

                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])

                    cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))

                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

                    cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
                    cv2.circle(output, (x2, y2), 6, (128, 0, 250), 4)
                    cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)

                    cv2.rectangle(output, (0, 0), (60, 60), (255, 255, 0), -1)

                    cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
                    cv2.putText(output, str(self.count), (10, 50), 1, 3.5, (128, 0, 250), 2)

                    cv2.imshow("output", output)

                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def save_state(self):
        return PoseStateMemento(self.up, self.down, self.count)

    def restore_state(self, memento):
        self.up = memento.up
        self.down = memento.down
        self.count = memento.count

class PoseDetectorCaretaker:
    def __init__(self):
        self.mementos = []

    def add_memento(self, memento):
        self.mementos.append(memento)

    def get_memento(self, index):
        return self.mementos[index]

# Uso del patrón Memento
if __name__ == "__main__":
    caretaker = PoseDetectorCaretaker()
    detector = PoseDetector("squats1.mp4")
    detector.start_detection()

    # Guardar el estado actual
    memento = detector.save_state()
    caretaker.add_memento(memento)

    # Restaurar el estado anterior
    old_memento = caretaker.get_memento(0)
    detector.restore_state(old_memento)
