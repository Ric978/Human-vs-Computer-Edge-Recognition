import cv2
import numpy as np

def canny_edge_detection(image_path, low_threshold, high_threshold):
    #Ottengo l'immagine tramite il path
    image = cv2.imread(image_path)

    if image is None:
        print("Impossibile leggere l'immagine. Assicurati che il percorso del file sia corretto.")
        return None

    # Conversione tramite la libreria cvtColor in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applicare una sfocatura gaussiana per ridurre il rumore
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calcolare il gradiente usando Sobel
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calcola la direzione del gradiente
    angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Soppressione dei punti non-massimi
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbor1, neighbor2 = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbor1, neighbor2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbor1, neighbor2 = magnitude[i-1, j], magnitude[i+1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                neighbor1, neighbor2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                suppressed[i, j] = magnitude[i, j]

    # Edge tracking by hysteresis => calcolo dei punti deboli, forti e non rilevanti in base alle soglie
    edges = np.zeros_like(suppressed)
    strong_edges = (suppressed > high_threshold)
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)

    edges[strong_edges] = 255
    edges[weak_edges] = 50

    # Utilizzo della funzione di soglia incorporata nella libreria OpenCV per una migliore visualizzazione dell'immagine
    _, result = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    return result

#Input: Immagine dipartimento unimi
input_image_path = 'dipartimentoInfo.jpg'
result_image = canny_edge_detection(input_image_path, low_threshold=50, high_threshold=150)

if result_image is not None:
    cv2.imshow('Immagine originale', cv2.imread(input_image_path))
    cv2.imshow('Rilevamento bordi Canny', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
