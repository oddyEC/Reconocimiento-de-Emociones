import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import time
import sys


def web_cam(face_detector,model,src=0,vid_rec = False):
    """
    Función para reconocer las emociones en tiempo real.
    Cambie src = 1, si está usando una cámara externa como
    fuente de imagen.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("No se puede iniciar la cámara")
        sys.exit(0)

    #Detección de rostros
    faceCascade = face_detector
    font = cv2.FONT_HERSHEY_SIMPLEX

    #Diccionario para la salida del modelo de reconocimiento de emociones y las emociones
    emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}

    #Subir imágenes de emojis desde la carpeta de emojis
    emoji = []
    #for index in range(6):
        #emotion = emotions[index]
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\enojado.jpeg'))
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\asustado.jpeg'))
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\feliz.jpeg'))
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\triste.jpeg'))
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\sorpresa.jpeg'))
    emoji.append(cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\neutral.jpeg'))


    #Código para grabar vídeo en tiempo real
    #Grabará vídeo si se indica explícitamente
    #Hacer vid_rec = True para grabar video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('emotion_rec1.avi', fourcc, 8.0, (640, 480))

    frame_count = 0

    while 1:
        # Captura frame a frame
        ret, frame = cap.read()

        #Comprobar si la imagen se está capturando desde la fuente
        if not ret:
            print("No image from source")
            break

        #Convertir la imagen RGB en gris para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #La detección de rostros tarda aproximadamente 0,07 segundos
        start_time = time.time()
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        #print("--- %s seconds ---" % (time.time() - start_time))

        #Escribiendo emociones en el frame
        y0 = 15
        for index in range(6):
            cv2.putText(frame, emotions[index] + ': ', (5, y0), font,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y0 += 15

        try:
            # Flag para mostrar el gráfico de probabilidad de una sola cara
            FIRSTFACE = True
            if len(faces) > 0:
                for x, y, width, height in faces:
                    cropped_face = gray[y:y + height,x:x + width]
                    test_image = cv2.resize(cropped_face, (48, 48))
                    test_image = test_image.reshape([-1,48,48,1])

                    test_image = np.multiply(test_image, 1.0 / 255.0)

                    #Probabilidades de todas las clases
                    #La búsqueda de la probabilidad de la clase tarda aproximadamente 0,05 segundos
                    start_time = time.time()
                    if frame_count % 5 == 0:
                        probab = model.predict(test_image)[0] * 100
                        #print("--- %s seconds ---" % (time.time() - start_time))

                        #Búsqueda de la etiqueta a partir de las probabilidades
                        #La clase con mayor probabilidad se considera etiqueta de salida
                        label = np.argmax(probab)
                        probab_predicted = int(probab[label])
                        predicted_emotion = emotions[label]
                        frame_count = 0

                    frame_count += 1
                    #Gráfica de probabilidad para la primera cara detectada
                    if FIRSTFACE:
                        y0 = 8
                        for score in probab.astype('int'):
                            cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                        font, 0.3, (0, 0, 255),1, cv2.LINE_AA)
                            cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                          (0, 255, 255), cv2.FILLED)
                            y0 += 15
                            FIRSTFACE =False

                    # Dibuja en el frame
                    font_size = width / 300
                    filled_rect_ht = int(height / 5)

                    #Cambiar el tamaño del emoji según el tamaño de la cara detectada
                    emoji_face = emoji[(label)]
                    emoji_face = cv2.resize(emoji_face, (filled_rect_ht, filled_rect_ht))

                    #Colocación de emojis en el frame
                    emoji_x1 = x + width - filled_rect_ht
                    emoji_x2 = emoji_x1 + filled_rect_ht
                    emoji_y1 = y + height
                    emoji_y2 = emoji_y1 + filled_rect_ht

                    #Dibujar el rectángulo y mostrar los valores de salida en el marco
                    cv2.rectangle(frame, (x, y), (x + width, y + height),(155,155, 0),2)
                    cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                                  (155, 155, 0),cv2.FILLED)
                    cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                                (x, y + height+ filled_rect_ht-10), font,font_size,(255,255,255), 1, cv2.LINE_AA)
                    video2= emotionImage(predicted_emotion)
                    cv2.imshow('Emoji',video2)
                    # Showing emoji on frame
                    for c in range(0, 3):
                        frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] = emoji_face[:, :, c] * \
                            (emoji_face[:, :, 3] / 255.0) + frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] * \
                            (1.0 - emoji_face[:, :, 3] / 255.0)

        except Exception as error:
            #print(error)
            pass


        cv2.imshow('frame', frame)

        if vid_rec:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()
def emotionImage(emotion):
    global imagen
    if emotion == 'Happy': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\feliz.jpeg')
    if emotion == 'Angry': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\enojado.jpeg')
    if emotion == 'Surprised': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\sorpresa.jpeg')
    if emotion == 'Sad': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\triste.jpeg')
    if emotion == 'Neutral': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\neutral.jpeg')
    if emotion == 'Fear': imagen = cv2.imread('C:\\Users\\diego\\Downloads\\Emojis\\asustado.jpeg')
    return imagen
def main():
    #Creating objects for face and emotiction detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    #emotion_model = load_model('fer.h5')
    #load model
    model = model_from_json(open("fer.json", "r").read())
    #load weights
    model.load_weights('fer.h5')
    web_cam(face_detector,model)

if __name__ == '__main__':
    main()
