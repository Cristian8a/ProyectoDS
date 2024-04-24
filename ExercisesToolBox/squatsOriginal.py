import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PoseStateMemento:
    '''Clase Memento para guardar el estado de la clase PoseDetector
    Crea un nuevo objeto para guardar el estado actual de la pose'''
    def __init__(self, up, down, count):
        self.up = up
        self.down = down
        self.count = count

class PoseDetector:
    '''Inicializa el detector de poses con el video proporcionado 
    y las dimensiones de la imagen'''
    def __init__(self, video_path, new_width=750, new_height=800):
        self.video_path = video_path
        self.new_width = new_width
        self.new_height = new_height
        self.up = False
        self.down = False
        self.count = 0
        self.mp_pose = mp.solutions.pose

    def start_detection(self):
        '''Comienza la detección de poses en el video proporcionado'''
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
                    if angle >= 140:
                        self.up = True
                    if self.up and not self.down and angle <= 70:
                        self.down = True
                    if self.up and self.down and angle >= 140:
                        self.count += 1
                        self.up = False
                        self.down = False

                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)

                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])

                    cv2.fillPoly(aux_image, pts=[contours], color=(0, 0, 255))

                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

                    cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
                    cv2.circle(output, (x2, y2), 6, (128, 0, 250), 4)
                    cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)

                    cv2.rectangle(output, (0, 0), (60, 60), (0, 0, 255), -1)

                    cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
                    cv2.putText(output, str(self.count), (10, 50), 1, 3.5, (255, 255, 0), 2)

                    cv2.imshow("output", output)

                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def save_state(self):
        '''Guarda el estado actual de detección'''
        return PoseStateMemento(self.up, self.down, self.count)

    def restore_state(self, memento):
        '''Restaura el estado de detección a un estado anterior'''
        self.up = memento.up
        self.down = memento.down
        self.count = memento.count

class PoseDetectorCaretaker:
    '''Inicializa el objeto Caretaker para guardar los mementos'''
    def __init__(self):
        self.mementos = []

    def add_memento(self, memento):
        '''Añade un memento a la lista de mementos'''
        self.mementos.append(memento)

    def get_memento(self, index):
        '''Obtiene un memento de la lista de mementos por indice'''
        return self.mementos[index]

# Uso del patrón Memento
if __name__ == "__main__":
    caretaker = PoseDetectorCaretaker()
    # detector = PoseDetector(cv2.VideoCapture(0,cv2.CAP_DSHOW))
    # detector = PoseDetector("SquatsRL.mp4")
    detector = PoseDetector("SquatsRL2.mp4")
    # detector = PoseDetector("squats2.mp4")
    detector.start_detection()

    # Guardar el estado actual
    memento = detector.save_state()
    caretaker.add_memento(memento)

    # Restaurar el estado anterior
    old_memento = caretaker.get_memento(0)
    detector.restore_state(old_memento)
