import pandas as pd

def clean_and_split_csv(input_csv):
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)

    # Asegurar que la columna 'Failure' tenga valores 0 o 1
    df['Failure'] = df['Failure'].fillna(0).astype(int)

    # Separar los datos por 'Type'
    device_data = df[df['Type'] == 'Devices'][['Timestamp', 'ID', 'BER', 'OSNR', 'Failure']]
    infrastructure_data = df[df['Type'] == 'Infrastructure'][['Timestamp', 'ID', 'InputPower', 'OutputPower', 'Failure']]

    # Guardar los datos de 'Device' en un CSV
    device_data.to_csv('Devices.csv', index=False)

    # Crear y guardar un archivo CSV para cada infraestructura
    infrastructure_ids = infrastructure_data['ID'].unique()
    for infra_id in infrastructure_ids:
        infra_subset = infrastructure_data[infrastructure_data['ID'] == infra_id]
        infra_subset.to_csv(f'{infra_id}.csv', index=False)

    print("Archivos generados con éxito: 'Device.csv' y archivos individuales para cada infraestructura.")

# Reemplaza 'input.csv' con el nombre de tu archivo original
input_csv = 'SoftFailure_dataset.csv'
clean_and_split_csv(input_csv)
