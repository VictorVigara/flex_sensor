import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dataframe_image as dfi

# Ruta a la carpeta de datos
data_folder_path = "/media/victor/DATA/rosbag_asta_data/25-07-manual-flight-collisions/tests_done/RNN_w40"

def parse_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        if "MAE" in line :
            metrics["MAE"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "MSE" in line and not "RMSE" in line:
            metrics["MSE"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "RMSE" in line:
            metrics["RMSE"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "StdDev" in line:
            metrics["StdDev"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "Median" in line:
            metrics["Median"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "IQR" in line:
            metrics["IQR"] = round(float(line.split('=')[-1].strip().split()[0]),4)
        elif "Max" in line:
            metrics["Max"] = round(float(line.split('=')[-1].strip().split()[0]),4)
    
    return metrics

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
        try:
            float(parts[1])
        except: 
            continue
        data.append(parts[:5])

    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    df = pd.DataFrame(data, columns=columns)
    df.set_index("Class", inplace=True)
    
    # Convert columns to the appropriate data types
    df = df.astype({"Precision": float, "Recall": float, "F1-Score": float, "Support": int})

    return df

# Función para leer el reporte de métricas de contacto
def read_contact_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        if "Contact Detection Accuracy" in line:
            metrics["Accuracy"] = round(float(line.split(':')[-1].strip()), 4)
        elif "Contact Detection Precision" in line:
            metrics["Precision"] = round(float(line.split(':')[-1].strip()), 4)
        elif "Contact Detection Recall" in line:
            metrics["Recall"] = round(float(line.split(':')[-1].strip()), 4)
        elif "Contact Detection F1 Score" in line:
            metrics["F1"] = round(float(line.split(':')[-1].strip()), 4)
        elif "Contact Detection AUC" in line:
            metrics["AUC"] = round(float(line.split(':')[-1].strip()), 4)
    
    return metrics

# Obtener las carpetas de los modelos
model_folders = [
    os.path.join(data_folder_path, f)
    for f in os.listdir(data_folder_path)
    if os.path.isdir(os.path.join(data_folder_path, f))
]

# Leer y almacenar los reportes de clasificación
orientation_reports = {}
displacement_reports = {}

# Initialize dictionaries to store metrics for all models
displacement_metrics = {}
orientation_metrics = {}
contact_metrics = {}

for model_folder in model_folders:
    orientation_report_path = os.path.join(model_folder, "classification_report_orientation.txt")
    displacement_report_path = os.path.join(model_folder, "classification_report_displacement.txt")
    contact_metrics_path = os.path.join(model_folder, "contact_analysis_metrics.txt")

    if os.path.exists(orientation_report_path):
        model_name = os.path.basename(model_folder)
        orientation_reports[model_name] = read_classification_report(orientation_report_path)
        orientation_metrics[model_name] = parse_metrics(orientation_report_path)

    if os.path.exists(displacement_report_path):
        model_name = os.path.basename(model_folder)
        displacement_reports[model_name] = read_classification_report(displacement_report_path)
        displacement_metrics[model_name] = parse_metrics(displacement_report_path)
    
    if os.path.exists(contact_metrics_path):
        model_name = os.path.basename(model_folder)
        contact_metrics[model_name] = read_contact_metrics(contact_metrics_path)

# Función para plotear comparaciones con subfiguras
def plot_comparisons_with_subplots(reports, title, output_path):
    # Apply background gradient
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
    os.path.join(data_folder_path, "orientation_detection_metrics.png"), 
)

# Graficar comparaciones para desplazamientos
plot_comparisons_with_subplots(
    displacement_reports,
    "Displacement Detection Metrics",
    os.path.join(data_folder_path, "displacement_detection_metrics.png"), 
)

# Convert the dictionaries to DataFrames
displacement_df = pd.DataFrame.from_dict(displacement_metrics, orient='index')
orientation_df = pd.DataFrame.from_dict(orientation_metrics, orient='index')
contact_df = pd.DataFrame.from_dict(contact_metrics, orient='index')

# Display and save the DataFrames
print("Displacement Metrics")
print(displacement_df)
print("Orientation Metrics")
print(orientation_df)
print("Contact Metrics")
print(contact_df)

# Save DataFrames to CSV
displacement_df.to_csv(os.path.join(data_folder_path, "displacement_metrics.csv"))
orientation_df.to_csv(os.path.join(data_folder_path, "orientation_metrics.csv"))
contact_df.to_csv(os.path.join(data_folder_path, "contact_metrics.csv"))

# Function to plot styled DataFrame
def plot_styled_dataframe(df, output_path, cmap="Greens_r"):
    # Apply background gradient
    styled_df = df.style.background_gradient(cmap=cmap)
    dfi.export(styled_df, output_path)

# Plot and save the styled DataFrames
plot_styled_dataframe(displacement_df, os.path.join(data_folder_path, "displacement_metrics_table.png"))
plot_styled_dataframe(orientation_df, os.path.join(data_folder_path, "orientation_metrics_table.png"))
plot_styled_dataframe(contact_df, os.path.join(data_folder_path, "contact_metrics_table.png"), cmap = "Greens")
