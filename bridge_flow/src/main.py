import requests
import hashlib
import os
import time
import random
import datetime
import gc

# URL de la imagen
url = "http://www.openlaredo.com/bridge/BridgeWebCamStills/bridge2MEX.jpg"

# Directorio para guardar las imágenes
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# Ruta para guardar la última imagen descargada
last_image_path = os.path.join(image_dir, "bridge_image_00000000_000000.jpg")

# Encabezados para evitar bloqueos por parte del servidor
user_agents = [
    'Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.43 Mobile Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.0; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

# Tiempo de espera base en segundos
min_wait = 30
max_wait = 60

# Límite de tamaño total de imágenes (1 GB)
max_total_size = 1 * 1024 * 1024 * 1024
# Límite de tamaño total de imágenes (1 MB)
# max_total_size = 4 * 1024 * 1024

def get_random_user_agent():
    return random.choice(user_agents)

def get_image_hash(image_content):
    """Obtiene el hash MD5 del contenido de una imagen."""
    return hashlib.md5(image_content).hexdigest()

def load_last_image_hash():
    """Carga el hash MD5 de la última imagen descargada."""
    if os.path.exists(last_image_path):
        with open(last_image_path, 'rb') as f:
            return get_image_hash(f.read())
    return None

def save_image(image_content):
    """Guarda la nueva imagen en el archivo y como una copia única."""
    # Guardar como última imagen
    with open(last_image_path, 'wb') as f:
        f.write(image_content)
    
    # Guardar como copia única con marca de tiempo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_image_path = os.path.join(image_dir, f"bridge_image_{timestamp}.jpg")
    with open(unique_image_path, 'wb') as f:
        f.write(image_content)

    print(f"Imagen guardada como {unique_image_path}")
    return unique_image_path
    

def get_total_directory_size(directory):
    """Calcula el tamaño total de los archivos en un directorio."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


# def clean_up_images(directory, max_images=100): # Limpiar imágenes viejas
#     images = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))])
#     if len(images) > max_images:
#         for img in images[:-max_images]:
#             os.remove(img)
#             print(f"Eliminando {img}")


def analyze_image(image_path):
    import torch
    from PIL import Image, ImageEnhance
    import cv2
    import os
    import matplotlib
    matplotlib.use('TkAgg')  # Configura el backend para Tkinter
    import matplotlib.pyplot as plt

    # Cargar el modelo preentrenado de YOLO (YOLOv5 en este caso)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cpu')  # Usar yolov5s (modelo ligero)

    # Configurar parámetros
    model.conf = 0.10 # Umbral de confianza
    model.iou = 0.45   # Umbral IoU
    model.max_det = 1000 # Número máximo de detecciones
    model.classes = [2, 7]  # [2, 5, 7] = ['car', 'bus', 'truck']

    # Ruta de la imagen
    image_path = os.path.join(unique_image_path)

    # Cargar la imagen
    image = Image.open(unique_image_path)
    image = ImageEnhance.Brightness(image).enhance(1.5)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    
    # Realizar la detección
    results = model(image_path, augment=True)

    # Extraer datos de detección
    detected_objects = results.pandas().xyxy[0]['name'].value_counts().to_dict()
    # print(results.pandas().xyxy[0]['name'].value_counts().to_dict())
    
    # Obtener la fecha y hora actuales
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_current_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.png'
    output_file = "detections_results.txt"
    with open(output_file, "a") as file:
        file.write(current_datetime + ": " + str(detected_objects) + "\n")

        results.render() 
        img_with_boxes = results.ims[0]
        # plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

        plt.figure(figsize=(16, 12), dpi=300) # Tamaño de la imagen 
        plt.imshow(
            cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), # Convertir de BGR a RGB 
            interpolation='bicubic', # Interpolación bicúbica
            aspect='auto' # Ajustar al tamaño de la imagen
        ) # Mostrar la imagen
        plt.tight_layout() # Ajustar el diseño
        plt.axis('off') # Desactivar los ejes
        plt.savefig(
            f'/home/admin/Desktop/img/{file_current_datetime}', 
            dpi=300, # Resolución de la imagen
            bbox_inches='tight', # Ajustar al contenido
        ) # Guardar la imagen
        print("Imagen guardada como: " + file_current_datetime)

    torch.cuda.empty_cache()
    del model
    del image
    del results
    gc.collect()

try:
    print("Presiona Ctrl+C para interrumpir el loop.")
    while True:
        try:
            # Verificar el tamaño total del directorio
            current_total_size = get_total_directory_size(image_dir)
            if current_total_size >= max_total_size:
                print("El tamaño total de las imágenes ha alcanzado el límite de 1 GB. Deteniendo el programa.")
                break

            # Encabezados aleatorios para evitar bloqueos
            headers = {
                'User-Agent': get_random_user_agent()
            }

            # Realizar la solicitud HTTP para descargar la imagen
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)

            if response.status_code == 200 and response.content:
                # Calcular el hash de la imagen descargada
                current_image_hash = get_image_hash(response.content)

                # Cargar el hash de la última imagen
                last_image_hash = load_last_image_hash()

                if current_image_hash != last_image_hash:
                    # Guardar la nueva imagen y avisar del cambio
                    unique_image_path = save_image(response.content)
                    print("La imagen ha cambiado. Nueva imagen descargada.")

                    # Analizar la imagen
                    try:
                        analyze_image(unique_image_path)
                    except Exception as e:
                        print(f"Error al analizar la imagen: {e}")
                        
                else:
                    print("La imagen no ha cambiado.")
            elif response.status_code == 429:
                print("Demasiadas solicitudes. Esperando antes de reintentar...")
                time.sleep(random.randint(60, 120))
            else:
                print(f"No se pudo descargar la imagen correctamente. Código de estado: {response.status_code}")

            # Esperar antes de la próxima iteración (intervalo aleatorio)
            wait_time = random.randint(min_wait, max_wait)
            print(f"Esperando {wait_time} segundos antes de la próxima solicitud...")
            time.sleep(wait_time)

        except requests.exceptions.SSLError:
            print("Error de SSL. Intentando sin verificar el certificado.")
            try:
                response = requests.get(url, headers=headers, timeout=10, verify=False)
                if response.status_code == 200:
                    save_image(response.content)
                    print("Imagen descargada sin verificación de certificado SSL.")
            except requests.exceptions.RequestException as e:
                print(f"Error al realizar la solicitud incluso sin SSL: {e}")

        except requests.exceptions.RequestException as e:
            print(f"Error al realizar la solicitud: {e}")

    import cv2
    import os

    # Parámetros del video
    output_file = "output_video.mp4"
    frame_rate = 30  # FPS
    image_folder = image_dir

    # Obtener la lista de imágenes
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ordenar por nombre

    # Leer la primera imagen para determinar el tamaño del video
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # Agregar las imágenes al video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Liberar el objeto VideoWriter
    video.release()

    print(f"Video creado: {output_file}")

except KeyboardInterrupt:
    print("Interrumpido por el usuario.")
