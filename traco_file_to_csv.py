import json
import csv
import os

"""
Function to convert JSON to CSV
Here t in the csv is z in the json. ID in the json is hexbug in the csv and pos in the json is split up to x and y in the csv
"""
def json_to_csv(json_folder, csv_folder):
    # Iterate over each JSON file in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.traco'):
            json_file_path = os.path.join(json_folder, filename)
            csv_file_path = os.path.join(csv_folder, filename.replace('.traco', '.csv'))
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                # Open CSV file for writing
                with open(csv_file_path, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    # Write headers to CSV file
                    writer.writerow(['', 't', 'hexbug', 'x', 'y'])
                    # Write data to CSV file
                    for idx, item in enumerate(data['rois']):
                        writer.writerow([idx, item['z'], item['id'], item['pos'][0], item['pos'][1]])

# Specify your JSON folder and CSV folder paths
json_folder = 'training'
csv_folder = 'training_CSV'

# Convert JSON files to CSV
json_to_csv(json_folder, csv_folder)