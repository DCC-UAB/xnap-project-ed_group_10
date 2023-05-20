import os
import json

def create_json_file(folder_path, json_file):
    data = {
        "train": [],
        "test": []
    }

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename not in ["all.txt", "train.txt", "test.txt"]:
            file_path = os.path.join(folder_path, filename)
            label = int(filename.split("_")[0]) - 1
            samples = []
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    sample_id = line.strip().split("\t")[0]
                    sample = [sample_id, label]
                    samples.append(sample)

            if "train" in filename:
                data["train"].extend(samples)
            elif "test" in filename:
                data["test"].extend(samples)

    with open(json_file, 'w') as file:
        json.dump(data, file)

# Ruta de la carpeta "0" y nombre del archivo JSON resultante
folder_path = "../features/data/ImageSets/0"
json_file = "../features/data/ImageSets/0/split_0.json"

# Crear el archivo JSON con los datos de los archivos de texto en la carpeta
create_json_file(folder_path, json_file)
