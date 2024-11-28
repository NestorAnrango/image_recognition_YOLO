from ultralytics import YOLO
import os
import matplotlib.pyplot as plt


# Obtener la ruta actual
cwd = os.getcwd()

# Definir la ruta de la imagen
image_path = os.path.join(cwd, 'data', 'car.PNG')


def predict(image_path):
    # Cargar el modelo YOLO preentrenado
    yolo = YOLO("yolov5s.pt")

    # Realizar la predicción
    results = yolo.predict(image_path)

    # Devolver los resultados
    return results


# Obtener los resultados de la predicción
res = predict(image_path)

# Visualize the results
for result in res:
    result.show()

# Visualizar los resultados
# res[0].plot()
res[0].save("./results/output.png")