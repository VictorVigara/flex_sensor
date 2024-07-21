import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ruta a la carpeta de datos
data_folder_path = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset"

# Función para leer el reporte de clasificación
def read_classification_report(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    start_line = lines.index("Classification Report:\n") + 1
    report_lines = lines[start_line:]

    data = []
    for line in report_lines:
        parts = line.split()
        if len(parts) < 5 or parts[0] in ["accuracy", "macro", "weighted", "MAE"]:
            continue
        data.append(parts[:5])

    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    df = pd.DataFrame(data, columns=columns)
    df.set_index("Class", inplace=True)
    
    # Convert columns to the appropriate data types
    df = df.astype({"Precision": float, "Recall": float, "F1-Score": float, "Support": int})

    return df

# Obtener las carpetas de los modelos
model_folders = [
    os.path.join(data_folder_path, f)
    for f in os.listdir(data_folder_path)
    if os.path.isdir(os.path.join(data_folder_path, f))
]

# Leer y almacenar los reportes de clasificación
orientation_reports = {}
displacement_reports = {}

for model_folder in model_folders:
    orientation_report_path = os.path.join(model_folder, "classification_report_orientation.txt")
    displacement_report_path = os.path.join(model_folder, "classification_report_displacement.txt")

    if os.path.exists(orientation_report_path):
        model_name = os.path.basename(model_folder)
        orientation_reports[model_name] = read_classification_report(orientation_report_path)

    if os.path.exists(displacement_report_path):
        model_name = os.path.basename(model_folder)
        displacement_reports[model_name] = read_classification_report(displacement_report_path)

# Función para plotear comparaciones con subfiguras
def plot_comparisons_with_subplots(reports, title, output_path):
    metrics = ["Precision", "Recall", "F1-Score"]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for ax, metric in zip(axs, metrics):
        for model, report in reports.items():
            ax.plot(report.index, report[metric], marker="o", label=model)
        ax.set_title(f"{metric} for {title}")
        ax.set_xlabel("Classes")
        ax.set_ylabel(metric)
        ax.legend(title='Models')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Graficar comparaciones para orientaciones
plot_comparisons_with_subplots(
    orientation_reports,
    "Orientation Detection Metrics",
    os.path.join(data_folder_path, "orientation_detection_metrics.png")
)

# Graficar comparaciones para desplazamientos
plot_comparisons_with_subplots(
    displacement_reports,
    "Displacement Detection Metrics",
    os.path.join(data_folder_path, "displacement_detection_metrics.png")
)
