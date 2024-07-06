import os
import re

import joblib
import numpy as np
import pandas as pd
from result_analysis import displacement_analysis, orientation_analysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# Función para cargar los datos desde archivos CSV
def load_data(data_folder_path):
    # Obtener lista de archivos CSV en el directorio especificado
    file_list = [f for f in os.listdir(data_folder_path) if f.endswith(".csv")]

    all_data = []
    all_orientations = []
    all_positions = []
    force_applied = []

    # Expresión regular para extraer orientación y posición desde el nombre del archivo
    orientation_pattern = re.compile(r"orientation_(\d+)_pos_(\d\.\d)")

    for file_name in file_list:
        match = orientation_pattern.search(file_name)
        if match:
            orientation = int(match.group(1))
            position = float(match.group(2))
            # Leer los datos desde el archivo CSV
            data = pd.read_csv(
                os.path.join(data_folder_path, file_name), header=None
            ).values

            all_data.append(data)
            all_orientations.append(np.full(data.shape[0], orientation))
            all_positions.append(np.full(data.shape[0], position))
            force_applied.append(np.full(data.shape[0], position != 0.0))

    # Concatenar todos los datos en matrices
    all_data = np.vstack(all_data)
    all_orientations = np.hstack(all_orientations)
    all_positions = np.hstack(all_positions)
    force_applied = np.hstack(force_applied)

    return all_data, all_orientations, all_positions, force_applied


def calculate_differences(data):
    # Calcular diferencias entre pares de lecturas de sensores
    diff_data = []
    num_sensors = data.shape[1]
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            diff_data.append(data[:, i] - data[:, j])
    return np.array(diff_data).T


if __name__ == "__main__":
    # Ruta al directorio de datos
    data_folder_path = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/04-07-8pos-5disp/"
    )
    # Guardar el scaler para su uso futuro
    model_type = "knn_differences"  # Cambiar a 'linear_raw', 'linear_differences', 'knn_raw', 'knn_differences'

    # Definir los centros y el ancho de los bins para orientaciones
    centers_orientation = [0, 45, 90, 135, 180, 225, 270, 315]

    # Definir los centros y el ancho de los bins para desplazamientos
    centers_displacement = [0.5, 1, 1.5, 2, 2.5]

    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    # Cargar todos los datos
    all_data, all_orientations, all_positions, force_applied = load_data(
        data_folder_path
    )

    # Seleccionar si se usan diferencias o datos originales
    if "differences" in model_type:
        all_data = calculate_differences(all_data)

    # Normalizar los datos utilizando StandardScaler
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    joblib.dump(scaler, os.path.join(model_output_path, "scaler.pkl"))

    # Normalizar las orientaciones a un rango de 0 a 1
    all_orientations_normalized = all_orientations / 360.0

    # Dividir los datos en conjuntos de entrenamiento y prueba
    (
        X_train,
        X_test,
        y_train_force,
        y_test_force,
        y_train_angle,
        y_test_angle,
        y_train_disp,
        y_test_disp,
    ) = train_test_split(
        all_data,
        force_applied,
        all_orientations_normalized,
        all_positions,
        test_size=0.2,
        random_state=42,
    )

    # Selección del modelo
    if "linear" in model_type:
        model_force = LinearRegression()
        model_angle = LinearRegression()
        model_disp = LinearRegression()
    elif "knn" in model_type:
        model_force = KNeighborsRegressor(n_neighbors=5)
        model_angle = KNeighborsRegressor(n_neighbors=5)
        model_disp = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError("Modelo no soportado: usa 'linear' o 'knn'")

    # Entrenar los modelos
    model_force.fit(X_train, y_train_force)
    model_angle.fit(X_train, y_train_angle)
    model_disp.fit(X_train, y_train_disp)

    # Predecir en el conjunto de prueba
    y_pred_force = model_force.predict(X_test)
    y_pred_angle = model_angle.predict(X_test)
    y_pred_disp = model_disp.predict(X_test)

    # Evaluar los modelos
    force_accuracy = accuracy_score(y_test_force, y_pred_force.round())
    angle_mae = mean_absolute_error(
        y_test_angle * 360.0, y_pred_angle * 360.0
    )  # Convertir de nuevo a grados
    disp_mae = mean_absolute_error(y_test_disp, y_pred_disp)

    print(f"Force Detection Accuracy: {force_accuracy:.2f}")
    print(f"Angle MAE: {angle_mae:.2f} degrees")
    print(f"Displacement MAE: {disp_mae:.2f} cm")

    # Guardar los modelos
    joblib.dump(model_force, os.path.join(model_output_path, "model_force.pkl"))
    joblib.dump(model_angle, os.path.join(model_output_path, "model_angle.pkl"))
    joblib.dump(model_disp, os.path.join(model_output_path, "model_disp.pkl"))

    # Convertir las orientaciones a grados
    y_test_angle_deg = y_test_angle * 360.0
    y_pred_angle_deg = y_pred_angle * 360.0

    orientation_analysis(
        y_test_angle_deg,
        y_pred_angle_deg,
        centers_orientation,
        model_output_path,
        model_type,
    )
    displacement_analysis(
        y_test_disp, y_pred_disp, centers_displacement, model_output_path, model_type
    )
