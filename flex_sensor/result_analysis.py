import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def bin_orientations(orientations, centers, bin_width):
    bin_edges = []
    for center in centers:
        bin_edges.append((center - bin_width / 2, center + bin_width / 2))
    binned_orientations = np.zeros_like(orientations, dtype=int)
    for i, (start, end) in enumerate(bin_edges):
        if start > end:
            mask = (orientations > start) | (orientations < end)
        else:
            mask = (orientations > start) & (orientations < end)
        binned_orientations[mask] = i
    return binned_orientations


def orientation_analysis(
    y_test_angle_deg,
    y_pred_angle_deg,
    centers_orientation,
    data_folder_path,
    model_type,
):
    # Crear carpeta para el modelo
    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    bin_width_orientation = abs(centers_orientation[1] - centers_orientation[0])

    # Binarizar las orientaciones
    y_test_angle_binned = bin_orientations(
        y_test_angle_deg, centers_orientation, bin_width_orientation
    )
    y_pred_angle_binned = bin_orientations(
        y_pred_angle_deg, centers_orientation, bin_width_orientation
    )

    # Calcular la matriz de confusión para orientaciones
    conf_matrix_angle = confusion_matrix(
        y_test_angle_binned,
        y_pred_angle_binned,
        labels=np.arange(len(centers_orientation)),
    )

    # Calcular el informe de clasificación para orientaciones
    class_report_angle = classification_report(
        y_test_angle_binned,
        y_pred_angle_binned,
        labels=np.arange(len(centers_orientation)),
        target_names=[
            f"{centers_orientation[i]}°±{bin_width_orientation/2}°"
            for i in range(len(centers_orientation))
        ],
    )

    print("Confusion Matrix for Orientation Prediction:")
    print(conf_matrix_angle)

    print("Classification Report for Orientation Prediction:")
    print(class_report_angle)

    # Guardar el informe de clasificación para orientaciones en un archivo
    with open(
        os.path.join(model_output_path, "classification_report_orientation.txt"), "w"
    ) as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix_angle))
        f.write("\n\nClassification Report:\n")
        f.write(class_report_angle)

    # Graficar la matriz de confusión para orientaciones
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix_angle,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            f"{centers_orientation[i]}°±{bin_width_orientation/2}°"
            for i in range(len(centers_orientation))
        ],
        yticklabels=[
            f"{centers_orientation[i]}°±{bin_width_orientation/2}°"
            for i in range(len(centers_orientation))
        ],
    )
    plt.xlabel("Predicted Orientation Bin")
    plt.ylabel("True Orientation Bin")
    plt.title("Confusion Matrix for Orientation Prediction")
    plt.savefig(os.path.join(model_output_path, "confusion_matrix_orientation.png"))
    plt.show()

    # Graficar el informe de clasificación para orientaciones
    report_data_angle = []
    lines = class_report_angle.split("\n")
    for line in lines[2 : 2 + len(centers_orientation)]:
        if line != "":
            row = {}
            row_data = line.split()
            try:
                row["class"] = row_data[0]
            except:
                pass
            row["precision"] = float(row_data[1])
            row["recall"] = float(row_data[2])
            row["f1_score"] = float(row_data[3])
            report_data_angle.append(row)
    df_report_angle = pd.DataFrame.from_dict(report_data_angle)
    df_report_angle.set_index("class", inplace=True)
    df_report_angle.plot(kind="bar", figsize=(10, 7))
    plt.title("Classification Report for Orientation")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.savefig(
        os.path.join(model_output_path, "classification_report_orientation.png")
    )
    plt.show()


def displacement_analysis(
    y_test_disp, y_pred_disp, centers_displacement, data_folder_path, model_type
):
    # Crear carpeta para el modelo
    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    bin_width_displacement = abs(centers_displacement[1] - centers_displacement[0])

    # Binarizar los desplazamientos
    y_test_disp_binned = bin_orientations(
        y_test_disp, centers_displacement, bin_width_displacement
    )
    y_pred_disp_binned = bin_orientations(
        y_pred_disp, centers_displacement, bin_width_displacement
    )

    # Calcular la matriz de confusión para desplazamientos
    conf_matrix_disp = confusion_matrix(
        y_test_disp_binned,
        y_pred_disp_binned,
        labels=np.arange(len(centers_displacement)),
    )

    # Calcular el informe de clasificación para desplazamientos
    class_report_disp = classification_report(
        y_test_disp_binned,
        y_pred_disp_binned,
        labels=np.arange(len(centers_displacement)),
        target_names=[
            f"{centers_displacement[i]}±{bin_width_displacement/2}"
            for i in range(len(centers_displacement))
        ],
    )

    print("Confusion Matrix for Displacement Prediction:")
    print(conf_matrix_disp)

    print("Classification Report for Displacement Prediction:")
    print(class_report_disp)

    # Guardar el informe de clasificación para desplazamientos en un archivo
    with open(
        os.path.join(model_output_path, "classification_report_displacement.txt"), "w"
    ) as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix_disp))
        f.write("\n\nClassification Report:\n")
        f.write(class_report_disp)

    # Graficar la matriz de confusión para desplazamientos
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix_disp,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            f"{centers_displacement[i]}±{bin_width_displacement/2}"
            for i in range(len(centers_displacement))
        ],
        yticklabels=[
            f"{centers_displacement[i]}±{bin_width_displacement/2}"
            for i in range(len(centers_displacement))
        ],
    )
    plt.xlabel("Predicted Displacement Bin")
    plt.ylabel("True Displacement Bin")
    plt.title("Confusion Matrix for Displacement Prediction")
    plt.savefig(os.path.join(model_output_path, "confusion_matrix_displacement.png"))
    plt.show()

    # Graficar el informe de clasificación para desplazamientos
    report_data_disp = []
    lines = class_report_disp.split("\n")
    for line in lines[2 : 2 + len(centers_displacement)]:
        if line != "":
            row = {}
            row_data = line.split()
            row["class"] = row_data[0]
            row["precision"] = float(row_data[1])
            row["recall"] = float(row_data[2])
            row["f1_score"] = float(row_data[3])
            report_data_disp.append(row)
    df_report_disp = pd.DataFrame.from_dict(report_data_disp)
    df_report_disp.set_index("class", inplace=True)
    df_report_disp.plot(kind="bar", figsize=(10, 7))
    plt.title("Classification Report for Displacement")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.savefig(
        os.path.join(model_output_path, "classification_report_displacement.png")
    )
    plt.show()
