import cv2
import mediapipe as mp

def distancia_euc(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d
# Inicializamos algunas variables para contar repeticiones de lagartijas.
start = 0 
cnt = 0

# Importamos algunas clases y métodos específicos de Mediapipe.
np_drawing = mp.solutions.drawing_utils
np_pose = mp.solutions.pose

# Abrimos el archivo de video o la cámara para capturar las imágenes.
cap = cv2.VideoCapture("PU_RL.mp4")

#if not cap.isOpened():
#    print("Error al abrir el archivo de video")
#    exit()

# Creamos una instancia del modelo de detección de pose de Mediapipe.
with np_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Capturamos cada cuadro de video.
        success, img = cap.read()
        if not success:
            print('Not Success')
            break

        # Redimensionar la imagen
        # Ajusta los valores de new_width y new_height según lo que necesites
        new_width = 640
        new_height = 480
        img = cv2.resize(img, (new_width, new_height))
        
        # Invertimos la imagen horizontalmente para que coincida con el espejo del vídeo
        img = cv2.flip(img, 1)

        # Procesamos la imagen con el modelo de detección de pose de Mediapipe
        ##img.flags.writeable = False
        results = pose.process(img)
        image_heigth, image_width, _ = img.shape

        # Si se detecta una pose en el cuadro actual.
        if results.pose_landmarks:
            # Extraemos las coordenadas de puntos específicos de la pose.
            leftWrist =     (int(results.pose_landmarks.landmark[15].x*image_width),
                            int(results.pose_landmarks.landmark[15].y*image_heigth))
            rigthWrist =    (int(results.pose_landmarks.landmark[16].x*image_width),
                            int(results.pose_landmarks.landmark[16].y*image_heigth))
            leftShoulder =  (int(results.pose_landmarks.landmark[11].x*image_width),
                            int(results.pose_landmarks.landmark[11].y*image_heigth))
            rigthShoulder = (int(results.pose_landmarks.landmark[12].x*image_width),
                             int(results.pose_landmarks.landmark[12].y*image_heigth))
            
            # Dibujamos círculos en las muñecas y hombros para referencia visual.
            cv2.circle(img, leftWrist, 6, (0,0,255), 15)
            cv2.circle(img, leftShoulder, 6, (0,0,255), 15)

            # Verificamos si se ha iniciado o finalizado una lagartija.
            if distancia_euc(leftShoulder, leftWrist) < 245:
                start = 1
            elif start and distancia_euc(leftShoulder, leftWrist) > 460:
                cnt += 1
                start = 0

            np_drawing.draw_landmarks(img, results.pose_landmarks, np_pose.POSE_CONNECTIONS)

            # Mostramos el contador de lagartijas en la imagen.
            cv2.putText(img, str(cnt), (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        # Mostramos la imagen en una ventana.
        cv2.imshow('image', img)

        # Esperamos a que se presione la tecla 'q' para salir del bucle.
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

# Liberamos los recursos y cerramos todas las ventanas al salir.
cap.release()
cv2.destroyAllWindows()

