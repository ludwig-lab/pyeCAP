import os
import json
from datetime import datetime
import glob

def create_code_snapshot(folder_path, extensions, max_lines=10000):
    def remove_old_snapshots(directory, exclude_file):
        for file in glob.glob(os.path.join(directory, 'code_snapshot_*.json')):
            if file != exclude_file:
                os.remove(file)

    def list_files(directory, extensions):
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Exclude .pyc files and code_snapshot JSON files
                if file.endswith('.pyc') or (file.startswith('code_snapshot') and file.endswith('.json')):
                    continue

                if any(file.endswith(ext) for ext in extensions):
                    yield os.path.join(root, file)


    def read_file(file_path, max_lines):
        encodings = ['utf-8', 'latin-1', 'windows-1252']  # Add other encodings as needed
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    lines = file.readlines()
                    truncated = len(lines) > max_lines
                    return lines[:max_lines], truncated
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read with 'utf-8' and replace errors
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
            truncated = len(lines) > max_lines
            return lines[:max_lines], truncated

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    json_filename = f'code_snapshot_{timestamp}.json'
    json_path = os.path.join(folder_path, json_filename)

    # Create and write new snapshot file
    data = []
    for file_path in list_files(folder_path, extensions):
        lines, truncated = read_file(file_path, max_lines)
        if truncated:
            lines.append('... (Content truncated)')
        data.append({
            'file_path': file_path,
            'content': lines
        })

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    # Remove old snapshot files, excluding the newly created one
    remove_old_snapshots(folder_path, json_path)

    print(f'JSON file created at: {json_path}')
