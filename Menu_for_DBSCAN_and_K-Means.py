import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time


def guardar_frames(video_path, output_dir, time_interval):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    frame_number = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time

        if elapsed_time >= time_interval:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            thresh = cv2.erode(thresh, None, iterations=3)
            thresh = cv2.dilate(thresh, None, iterations=3)

            black_canvas = np.zeros_like(gray2)

            black_canvas[thresh > 0] = gray2[thresh > 0]

            motion_display = cv2.cvtColor(black_canvas, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(f"{output_dir}/frame_{frame_number:04d}.png", black_canvas)

            resized_video = cv2.resize(motion_display, (800, 600))
            cv2.imshow("Processed Video", resized_video)

            gray1 = gray2
            start_time = time.time()

            frame_number += 1
            print(f"frame # {frame_number}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def suma_imagenes(folder_path, output_image_path):
    summed_matrix = None
    count = 1

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = mpimg.imread(img_path)
            if img.ndim == 3:
                img = img[:, :, 0]
            binary_img = (img > 0.5).astype(int) 
            
            if summed_matrix is None: 
                summed_matrix = np.zeros_like(binary_img, dtype=int)
            
            summed_matrix += binary_img
            print(f"imagen #{count}")
            count += 1

    print(summed_matrix)

    plt.imshow(summed_matrix, cmap='gray')
    plt.show()

    plt.imsave(output_image_path, summed_matrix, cmap='gray')

def marcar_clusters(image_path, dbscan):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.flip(image,0)

    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    white_points = np.column_stack(np.where(binary_image == 255))

    if(dbscan):
        db = DBSCAN(eps=5, min_samples=10).fit(white_points)
    else:
        n_clusters = int(input("Ingrese el número de clusters: \n"))
        db = KMeans(n_clusters=n_clusters, random_state=10).fit(white_points)
    
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if(dbscan):
        print(f'Número de clusters: {n_clusters}')
        print(f'Número de puntos de ruido: {n_noise}')

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (labels == k)
        xy = white_points[class_member_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

        cluster_center = np.mean(xy, axis=0)
        cluster_radius = np.linalg.norm(xy - cluster_center, axis=1).max()
        circle = plt.Circle((cluster_center[1], cluster_center[0]), cluster_radius, color=tuple(col), fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    if(dbscan):
        plt.title(f'Clusters con DBSCAN: {n_clusters}')
    else:
        plt.title(f'Clusters con K-Means: {n_clusters}')
    plt.gca().invert_yaxis()
    plt.show()
    

print("Clustering de zonas más concurridas a partir de videos")

video_path = 'videos/HDV_0802.mp4'
output_dir = 'borrar'
time_interval = 10

while True:
    os.system('cls')
    print("Opciones:")
    print("1- Salir.")
    print("2- Guardar Frames.")
    print("3- Sumar Frames.")
    print("4- Clustering.")
    op = input("Selecciona una opción: ")

    if op == "1":
        os.system('cls')
        print("Saliendo.")
        break
    elif op == "2":
        os.system('cls')
        video_path = input("Ingrese la dirección de su video: \n")
        output_dir = input("Ingrese el destino de las imágenes a guardar: \n")
        time_interval = int(input("Ingrese el intervalo de tiempo (en segundos): \n"))

        try:
            guardar_frames(video_path, output_dir, time_interval)
        except Exception:
            print("Error al cargar la funcion")
    elif op == "3":
        os.system('cls')
        print("Usar ruta guardada?")
        print("1- Si.")
        print("2- No.")
            
        while True:
            op2 = input("Selecciona una opción: ")

            if op2 == "1":
                break
            elif op2 == "2":
                output_dir = input("Ingrese la nueva ruta del folder: \n")
                break
            else:
                print("Opción no válida, por favor intenta de nuevo.")

        try:
            print("Suma de matrices de las imagenes")
            output_image_path = os.path.join(output_dir, "suma_imagenes.png")
            suma_imagenes(output_dir, output_image_path)
        except Exception:
            print("Error al cargar la funcion")
    elif op == "4":
        os.system('cls')
        print("Usar ruta guardada?")
        print("1- Si.")
        print("2- No.")
        
        while True:
            op2 = input("Selecciona una opción: ")

            if op2 == "1":
                break
            elif op2 == "2":
                output_image_path = input("Ingrese la nueva ruta de la suma: \n")
                break
            else:
                print("Opción no válida, por favor intenta de nuevo.")
        
        os.system('cls')
        print("Seleccione el metodo a usar:")
        print("1- DBSCAN.")
        print("2- K-Means.")
        
        while True:
            op3 = input("Selecciona una opción: ")

            if op3 == "1":
                marcar_clusters(output_image_path, True)
                break
            elif op3 == "2":
                marcar_clusters(output_image_path, False)
                break
            else:
                print("Opción no válida, por favor intenta de nuevo.")
    else:
        print("Opción no válida, por favor intenta de nuevo.")
