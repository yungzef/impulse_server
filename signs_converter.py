import os
import subprocess

input_folder = '/Users/serhiifilatov/Downloads/signs_road'
output_folder = 'signs_road_png'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.svg'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        try:
            subprocess.run([
                '/Applications/Inkscape.app/Contents/MacOS/inkscape',  # путь до inkscape на macOS
                input_path,
                '--export-type=png',
                f'--export-filename={output_path}'
            ], check=True)
            print(f'✅ {filename} → PNG сохранён')
        except subprocess.CalledProcessError as e:
            print(f'❌ Ошибка при конвертации {filename}: {e}')

print('\n🎉 Готово: все SVG → PNG через Inkscape')
