import os

folders = [
    'templates',
    'static',
]

files = {
    'templates/home.html': '',
    'static/home.css': '',
    'app.py': '',
    'requirements.txt': '',
    '.env': '',
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f'Created folder: {folder}')

for file, content in files.items():
    with open(file, 'w') as f:
        f.write(content)
    print(f'Created file: {file}')

print('Project structure ready!')