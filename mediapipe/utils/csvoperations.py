import os
import csv

def save_data(row_id, data, filepath):
    dirpath = os.path.dirname(filepath)
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Directory '{dirpath}' does not exist.")
    if not os.access(dirpath, os.W_OK):
        raise PermissionError(f"No write access to directory '{dirpath}'.")
    
    with open(filepath, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([row_id, *data])
        
    return 0


def read_labels(filepath):
    dirpath = os.path.dirname(filepath)
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Directory '{dirpath}' does not exist.")
    if not os.access(dirpath, os.R_OK):
        raise PermissionError(f"No read access to directory '{dirpath}'.")
    
    with open(filepath, 'r') as csvfile:
        data = csv.reader(csvfile)
        data = list(data)
    
    labels = { int(id):gesture for id, gesture in data }
    
    return labels
