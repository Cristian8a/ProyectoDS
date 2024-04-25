import cv2
import mediapipe as mp
import numpy as np
import copy

class PoseStateMemento:
    '''Clase Memento para guardar el estado de la clase PoseDetector
    Crea un nuevo objeto para guardar el estado actual de la pose'''
    def __init__(self, start, cnt):
        self.start = start
        self.cnt = cnt

class PoseDetector:
    '''Inicializa el detector de poses con el video proporcionado 
    y las dimensiones de la imagen'''
    def __init__(self, video_path):
        self.video_path = video_path
        self.start = 0 
        self.cnt = 0
        self.np_drawing = mp.solutions.drawing_utils
        self.np_pose = mp.solutions.pose

    def start_detection(self):
        '''Comienza la detección de poses en el video proporcionado'''
        cap = cv2.VideoCapture(self.video_path)
        with self.np_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                # print('Inicio')
                success, frame = cap.read()
                if not success:
                    print('Not Success')
                    break

                # Hacer una copia modificable de la imagen
                img = copy.deepcopy(frame)

                # Procesar la imagen con Mediapipe
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_heigth, image_width, _ = img.shape

                if results.pose_landmarks:
                    leftWrist =     (int(results.pose_landmarks.landmark[15].x*image_width),
                                    int(results.pose_landmarks.landmark[15].y*image_heigth))
                    rightWrist =    (int(results.pose_landmarks.landmark[16].x*image_width),
                                    int(results.pose_landmarks.landmark[16].y*image_heigth))
                    leftShoulder =  (int(results.pose_landmarks.landmark[11].x*image_width),
                                    int(results.pose_landmarks.landmark[11].y*image_heigth))
                    rightShoulder = (int(results.pose_landmarks.landmark[12].x*image_width),
                                    int(results.pose_landmarks.landmark[12].y*image_heigth))
        
                    if self.distancia_euc(rightShoulder, rightWrist) < 245:
                        self.start = 1
                    elif self.start and self.distancia_euc(rightShoulder, rightWrist) > 460:
                        self.cnt += 1
                        self.start = 0

                    self.np_drawing.draw_landmarks(img, results.pose_landmarks, self.np_pose.POSE_CONNECTIONS)  # Corregir aquí

                    # Dibujar círculos más pequeños
                    cv2.circle(img, rightWrist, 6, (0,255,255), 10)
                    cv2.circle(img, rightShoulder, 6, (128,0,255), 10)
                    cv2.circle(img, leftWrist, 6, (255,191,0), 10)  
                    cv2.circle(img, leftShoulder, 6, (0,0,255), 10)

                    # Dibujar líneas auxiliares entre cada círculo
                    cv2.line(img, rightWrist, rightShoulder, (0, 255, 0), 3)
                    cv2.line(img, rightShoulder, leftShoulder, (0, 255, 0), 3)
                    cv2.line(img, leftShoulder, leftWrist, (0, 255, 0), 3)

                    cv2.rectangle(img, (0, 0), (60, 60), (0, 0, 255), -1)

                    # cv2.putText(img, str(self.cnt), (50, 100),
                    #             cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, str(self.cnt), (10, 50), 1, 3.5, (255, 255, 0), 2)

                    # Especificar el tamaño de la ventana
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('image', 900, 700)  # Ajusta el tamaño según tus necesidades

                    cv2.imshow('image', img)

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def save_state(self):
        '''Guarda el estado actual de detección'''
        return PoseStateMemento(self.start, self.cnt)

    def restore_state(self, memento):
        '''Restaura el estado de detección a un estado anterior'''
        self.start = memento.start
        self.cnt = memento.cnt

    def distancia_euc(self, p1, p2):
        '''Se utiliza la formula de Distancia Euclidiana para 
        calcular la distancia entre dos puntos'''
        d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return d

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
if  __name__ == "__main__":
    caretaker = PoseDetectorCaretaker()
    detector = PoseDetector("PU_RL.mp4")
    detector.start_detection()

    # Guardar el estado actual
    memento = detector.save_state()
    caretaker.add_memento(memento)

    # Restaurar el estado anterior
    old_memento = caretaker.get_memento(0)
    detector.restore_state(old_memento)
