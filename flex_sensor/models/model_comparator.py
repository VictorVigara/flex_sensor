import os

import matplotlib.pyplot as plt
import pandas as pd

# Ruta a la carpeta de datos
data_folder_path = (
    "/home/victor/ws_sensor_combined/src/flex_sensor/data/04-07-8pos-5disp/"
)

# Función para leer el reporte de clasificación
def read_classification_report(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    start_line = lines.index("Classification Report:\n") + 1
    report_lines = lines[start_line:]

    data = []
    for line in report_lines:
        if line.strip() == "":
            continue
        parts = line.split()
        if parts[0] == "accuracy":
            break
        data.append(parts)

    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    df = pd.DataFrame(data, columns=columns)
    df.set_index("Class", inplace=True)
    df = df.astype(
        {"Precision": float, "Recall": float, "F1-Score": float, "Support": int}
    )

    return df


# Obtener las carpetas de los modelos
model_folders = [
    f
    for f in os.listdir(data_folder_path)
    if os.path.isdir(os.path.join(data_folder_path, f))
]

# Leer y almacenar los reportes de clasificación
orientation_reports = {}
displacement_reports = {}

for model in model_folders:
    orientation_report_path = os.path.join(
        data_folder_path, model, "classification_report_orientation.txt"
    )
    displacement_report_path = os.path.join(
        data_folder_path, model, "classification_report_displacement.txt"
    )

    if os.path.exists(orientation_report_path):
        orientation_reports[model] = read_classification_report(orientation_report_path)

    if os.path.exists(displacement_report_path):
        displacement_reports[model] = read_classification_report(
            displacement_report_path
        )

# Función para plotear comparaciones
def plot_comparisons(reports, metric, title):
    plt.figure(figsize=(12, 8))

    for model, report in reports.items():
        plt.plot(report.index, report[metric], marker="o", label=model)

    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()


# Graficar comparaciones
plot_comparisons(
    orientation_reports,
    "Precision",
    "Comparison of Precision for Orientation Detection",
)
plot_comparisons(
    orientation_reports, "Recall", "Comparison of Recall for Orientation Detection"
)
plot_comparisons(
    orientation_reports, "F1-Score", "Comparison of F1-Score for Orientation Detection"
)

plot_comparisons(
    displacement_reports,
    "Precision",
    "Comparison of Precision for Displacement Detection",
)
plot_comparisons(
    displacement_reports, "Recall", "Comparison of Recall for Displacement Detection"
)
plot_comparisons(
    displacement_reports,
    "F1-Score",
    "Comparison of F1-Score for Displacement Detection",
)
