import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

def save_time_series_comparison_error(s1, s2, s3, s4, y_true, y_pred, errors, output_dir, plot_type):
    os.makedirs(output_dir, exist_ok=True)

    # Create the figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(s1, label="S1", linestyle="-")
    axs[0].plot(s2, label="S2", linestyle="-")
    axs[0].plot(s3, label="S3", linestyle="-")
    axs[0].plot(s4, label="S4", linestyle="-")
    axs[0].set_xlabel("Sample idx")
    axs[0].set_ylabel(f"ADC value")
    axs[0].set_title(f"Flex sensors output")
    axs[0].legend()
    axs[0].grid(True)

    # True vs. Predicted Plot
    axs[1].plot(y_true, label="True", linestyle="--")
    axs[1].plot(y_pred, label="Predicted", linestyle="-", alpha=0.75)
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel(f"{plot_type} ({'degrees' if plot_type == 'angle' else 'cm'})")
    axs[1].set_title(f"True vs. Predicted {plot_type.capitalize()} Over Samples")
    axs[1].legend()
    axs[1].grid(True)

    # Error Plot
    axs[2].plot(errors, label="Error", linestyle="--", color="red")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel(f"{plot_type.capitalize()} Error ({'degrees' if plot_type == 'angle' else 'cm'})")
    axs[2].set_title(f"{plot_type.capitalize()} Error Over Samples")
    axs[2].legend()
    axs[2].grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_type}_eval_flight_error_analysis.png"))
    plt.close()

def save_contact_event_plots(y_true, y_pred, errors, contact_idx, true_angle_or_disp, output_dir,s1, s2, s3, s4, plot_type):
    contact_event_folder = output_dir
    os.makedirs(contact_event_folder, exist_ok=True)

    # Create the figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(s1, label="S1", linestyle="-")
    axs[0].plot(s2, label="S2", linestyle="-")
    axs[0].plot(s3, label="S3", linestyle="-")
    axs[0].plot(s4, label="S4", linestyle="-")
    axs[0].set_xlabel("Sample idx")
    axs[0].set_ylabel(f"ADC value")
    axs[0].set_title(f"Flex sensors output")
    axs[0].legend()
    axs[0].grid(True)

    # True vs. Predicted Plot
    axs[1].plot(y_true, label="True", linestyle="--")
    axs[1].plot(y_pred, label="Predicted", linestyle="-", alpha=0.75)
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel(f"{plot_type} ({'degrees' if plot_type == 'angle' else 'cm'})")
    axs[1].set_title(f"True vs. Predicted {plot_type.capitalize()} Over Samples")
    axs[1].legend()
    axs[1].grid(True)

    # Error Plot
    axs[2].plot(errors, label="Error", linestyle="--", color="red")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel(f"{plot_type.capitalize()} Error ({'degrees' if plot_type == 'angle' else 'cm'})")
    axs[2].set_title(f"{plot_type.capitalize()} Error Over Samples")
    axs[2].legend()
    axs[2].grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(contact_event_folder, f"{plot_type}_contact_{contact_idx}_{plot_type}_{true_angle_or_disp}.png"))
    plt.close()


def calculate_angle_error(angulo1, angulo2):
    # Convertir los ángulos a radianes
    angulo1 = np.deg2rad(angulo1)
    angulo2 = np.deg2rad(angulo2)

    # Calcular la diferencia en radianes
    diff = angulo1 - angulo2

    # Ajustar la diferencia para que esté en el rango [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    # Convertir la diferencia de vuelta a grados
    diff = np.rad2deg(diff)

    return np.abs(diff)

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
    contact, 
    raw_values
):  
    y_test_angle_contact = y_test_angle_deg[contact==1]
    y_pred_angle_contact = y_pred_angle_deg[contact==1]

    # Crear carpeta para el modelo
    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)
    contact_events_folder = os.path.join(model_output_path, "contact_events")
    os.makedirs(contact_events_folder, exist_ok=True)

    angle_errors_contact = []
    angle_errors_all = []
    y_pred_angle_all_contact = []
    y_test_angle_all_contact = []
    contact_event_idx = 0
    in_contact_event = False
    current_contact_y_true = []
    current_contact_y_pred = []
    current_contact_errors = []
    current_sensor1 = []
    current_sensor2 = []
    current_sensor3 = []
    current_sensor4 = []

    for idx in range(len(y_test_angle_deg)):
        error = calculate_angle_error(y_test_angle_deg[idx].item(), y_pred_angle_deg[idx].item())

        if contact[idx] == 1.0:
            angle_errors_contact.append(error)
            angle_errors_all.append(error)
            y_pred_angle_all_contact.append(y_pred_angle_deg[idx].item())
            y_test_angle_all_contact.append(y_test_angle_deg[idx].item())

            current_contact_y_true.append(y_test_angle_deg[idx].item())
            current_contact_y_pred.append(y_pred_angle_deg[idx].item())
            current_contact_errors.append(error)
            current_sensor1.append(raw_values[idx][0])
            current_sensor2.append(raw_values[idx][1])
            current_sensor3.append(raw_values[idx][2])
            current_sensor4.append(raw_values[idx][3])


            in_contact_event = True
        else:
            angle_errors_all.append(0.0)
            y_pred_angle_all_contact.append(0.0)
            y_test_angle_all_contact.append(0.0)
            
            if in_contact_event:
                if len(current_contact_y_true) > 3:
                    save_contact_event_plots(
                        current_contact_y_true,
                        current_contact_y_pred,
                        current_contact_errors,
                        contact_event_idx,
                        y_test_angle_deg[idx - 1].item(),
                        contact_events_folder,
                        current_sensor1, 
                        current_sensor2, 
                        current_sensor3, 
                        current_sensor4,
                        "angle"
                    )
                contact_event_idx += 1
                current_contact_y_true = []
                current_contact_y_pred = []
                current_contact_errors = []
                current_sensor1 = []
                current_sensor2 = [] 
                current_sensor3 = [] 
                current_sensor4 = []
                in_contact_event = False

    angle_mae = np.mean(np.abs(angle_errors_contact))

    bin_width_orientation = abs(centers_orientation[1] - centers_orientation[0])

    # Binarizar las orientaciones
    y_test_angle_binned = bin_orientations(
        y_test_angle_contact, centers_orientation, bin_width_orientation
    )
    y_pred_angle_binned = bin_orientations(
        y_pred_angle_contact, centers_orientation, bin_width_orientation
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

    # Standard Deviation of the errors
    angle_std = np.std(angle_errors_contact)

    # Median Absolute Error
    angle_median = np.median(np.abs(angle_errors_contact))

    # Interquartile Range (IQR)
    q1 = np.percentile(angle_errors_contact, 25)
    q3 = np.percentile(angle_errors_contact, 75)
    angle_iqr = q3 - q1

    # Maximum Error
    angle_max = np.max(np.abs(angle_errors_contact))

    # Print the metrics
    print(f"Angle MAE: {angle_mae:.2f} degrees")
    print(f"Angle Std Dev: {angle_std:.2f} degrees")
    print(f"Angle Median: {angle_median:.2f} degrees")
    print(f"Angle IQR: {angle_iqr:.2f} degrees")
    print(f"Angle Max: {angle_max:.2f} degrees")

    # Guardar el informe de clasificación para orientaciones en un archivo
    with open(
        os.path.join(model_output_path, "classification_report_orientation.txt"), "w"
    ) as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix_angle))
        f.write("\n\nClassification Report:\n")
        f.write(class_report_angle)
        f.write(f"\n\nMAE angle = {angle_mae} deg")

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

    # Scatter Plot: True vs. Predicted Angles
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_angle_contact, y_pred_angle_contact, alpha=0.5)
    plt.plot([0, 360], [0, 360], color="red", linestyle="--")
    plt.xlabel("True Angles (degrees)")
    plt.ylabel("Predicted Angles (degrees)")
    plt.title("True vs. Predicted Angles")
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "scatter_plot_orientation.png"))
    plt.show()

    # Error Distribution Plot
    """ error_angles = y_test_angle_deg - y_pred_angle_deg
    error_angles[contact==0.0] = 0.0 """
    plt.figure(figsize=(10, 6))
    plt.hist(angle_errors_contact, bins=50, alpha=0.75, edgecolor="black")
    plt.xlabel("Prediction Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Angle Prediction Error Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "error_distribution_orientation.png"))
    plt.show()

    # Error timeseries Plot
    """ error_angles = y_test_angle_deg - y_pred_angle_deg
    error_angles[contact==0.0] = 0.0 """
    plt.figure(figsize=(10, 6))
    plt.plot(angle_errors_all, label="True Angles", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Angle (deg)")
    plt.title("Angle error timeseries")
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "angle_error_timeseries.png"))
    plt.show()

    # Time Series Plot
    plt.figure(figsize=(14, 8))
    plt.plot(y_test_angle_all_contact, label="True Angles", linestyle="--")
    plt.plot(
        y_pred_angle_all_contact, label="Predicted Angles", linestyle="-", alpha=0.75
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Angles (degrees)")
    plt.title("True vs. Predicted Angles Over Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "time_series_orientation.png"))
    plt.show()

    save_time_series_comparison_error(raw_values[:,0], raw_values[:,1], raw_values[:,2], raw_values[:,3], y_test_angle_deg, y_pred_angle_all_contact, angle_errors_all, model_output_path, "angle")


def displacement_analysis(
    y_test_disp, y_pred_disp, centers_displacement, data_folder_path, model_type, contact, raw_values
):
    y_test_disp_contact = y_test_disp[contact==1]
    y_pred_disp_contact = y_pred_disp[contact==1]
    # Crear carpeta para el modelo
    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)
    contact_events_folder = os.path.join(model_output_path, "contact_events")
    os.makedirs(contact_events_folder, exist_ok=True)

    disp_errors_contact = []
    disp_errors_all = []
    y_pred_disp_all_contact = []
    contact_event_idx = 0
    in_contact_event = False
    current_contact_y_true = []
    current_contact_y_pred = []
    current_contact_errors = []
    current_sensor1 = []
    current_sensor2 = []
    current_sensor3 = []
    current_sensor4 = []

    for idx in range(len(y_test_disp)):
        error = y_pred_disp[idx].item() - y_test_disp[idx].item()

        if contact[idx] == 1.0:
            disp_errors_contact.append(error)
            disp_errors_all.append(error)
            y_pred_disp_all_contact.append(y_pred_disp[idx].item())

            current_contact_y_true.append(y_test_disp[idx].item())
            current_contact_y_pred.append(y_pred_disp[idx].item())
            current_contact_errors.append(error)

            current_sensor1.append(raw_values[idx][0])
            current_sensor2.append(raw_values[idx][1])
            current_sensor3.append(raw_values[idx][2])
            current_sensor4.append(raw_values[idx][3])

            in_contact_event = True
        else:
            disp_errors_all.append(0.0)
            y_pred_disp_all_contact.append(0.0)

            if in_contact_event:
                if len(current_contact_y_true) > 3:
                    save_contact_event_plots(
                            current_contact_y_true,
                            current_contact_y_pred,
                            current_contact_errors,
                            contact_event_idx,
                            y_test_disp[idx - 1].item(),
                            contact_events_folder,
                            current_sensor1, 
                            current_sensor2, 
                            current_sensor3, 
                            current_sensor4,
                            "displacement"
                        )
                    contact_event_idx += 1
                current_contact_y_true = []
                current_contact_y_pred = []
                current_contact_errors = []
                current_sensor1 = []
                current_sensor2 = []
                current_sensor3 = []
                current_sensor4 = []
                disp_error_contact_filtered_temp = []
                in_contact_event = False



    disp_mae = np.mean(np.abs(disp_errors_contact))

    bin_width_displacement = abs(centers_displacement[1] - centers_displacement[0])

    # Binarizar los desplazamientos
    y_test_disp_binned = bin_orientations(
        y_test_disp_contact, centers_displacement, bin_width_displacement
    )
    y_pred_disp_binned = bin_orientations(
        y_pred_disp_contact, centers_displacement, bin_width_displacement
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

    print(f"MAE displacement = {disp_mae} cm")

    # Guardar el informe de clasificación para desplazamientos en un archivo
    with open(
        os.path.join(model_output_path, "classification_report_displacement.txt"), "w"
    ) as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix_disp))
        f.write("\n\nClassification Report:\n")
        f.write(class_report_disp)
        f.write(f"\n\nMAE displacement = {disp_mae} cms")

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

    # Scatter Plot: True vs. Predicted Displacements
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_disp_contact, y_pred_disp_contact, alpha=0.5)
    plt.plot([0, max(y_test_disp_contact)], [0, max(y_test_disp_contact)], color="red", linestyle="--")
    plt.xlabel("True Displacements (cm)")
    plt.ylabel("Predicted Displacements (cm)")
    plt.title("True vs. Predicted Displacements")
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "scatter_plot_displacement.png"))
    plt.show()

    # Error Distribution Plot
    error_displacements = y_pred_disp_contact - y_test_disp_contact
    plt.figure(figsize=(10, 6))
    plt.hist(disp_errors_contact, bins=50, alpha=0.75, edgecolor="black")
    plt.xlabel("Prediction Error (cm)")
    plt.ylabel("Frequency")
    plt.title("Displacement Prediction Error Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "error_distribution_displacement.png"))
    plt.show()

    # Time Series Plot
    plt.figure(figsize=(14, 8))
    plt.plot(y_test_disp, label="True Displacements", linestyle="--")
    plt.plot(
        y_pred_disp_all_contact, label="Predicted Displacements", linestyle="-", alpha=0.75
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Displacements (cm)")
    plt.title("True vs. Predicted Displacements Over Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, "time_series_displacement.png"))
    plt.show()

    save_time_series_comparison_error(raw_values[:,0], raw_values[:,1], raw_values[:,2], raw_values[:,3], y_test_disp, y_pred_disp_all_contact, disp_errors_all, model_output_path, "displacement")

