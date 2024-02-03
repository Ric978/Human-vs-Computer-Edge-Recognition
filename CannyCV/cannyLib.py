import cv2
 
# Carica l'immagine in scala di grigi
img = cv2.imread('dipartimentoInfo.jpg', cv2.IMREAD_GRAYSCALE)
 
# Verifica della presenza dell'immagine
if img is None:
    print('Impossibile caricare l\'immagine. Verifica il percorso del file.')
else:
    # Applico l'algoritmo di Canny tramite la libreria e la funzione Canny
    edges = cv2.Canny(img, 100, 200)
 
    # Mostra l'immagine originale e quella con i bordi rilevati
    cv2.imshow('Original Image', img)
    cv2.imshow('Edge Image', edges)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()