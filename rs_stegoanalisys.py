from PIL import Image
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ===================== НАСТРОЙКИ =====================
base_folder = r"C:\Users\Alexey\Desktop\Desktop"

folders = {
    "50 байт":  "50bait(1)",
    "100 байт": "100bait(1)",
    "500 байт": "500bait(1)",
    "1000 байт":"1000bait(1234)"
}

max_images_per_folder = 1000
group_size = 4
workers = 8
# ====================================================

def rs_analysis_image(image_path):
    try:
        # Размер файла на диске в КБ
        file_size_kb = Path(image_path).stat().st_size / 1024
        
        img = Image.open(image_path).convert('L')
        pixels = np.array(img, dtype=np.int16)
        h, w = pixels.shape
        
        mask = np.array([0, 1, 1, 0], dtype=np.int16)
        R = S = 0
        total = 0
        
        for y in range(h):
            row = pixels[y]
            for x in range(0, w - group_size + 1, group_size):
                block = row[x:x+group_size]
                if len(block) < group_size: 
                    continue
                f_orig = np.sum(np.abs(np.diff(block)))
                modified = block.copy()
                modified[mask == 1] ^= 1
                f_mod = np.sum(np.abs(np.diff(modified)))
                
                if f_mod > f_orig: R += 1
                elif f_mod < f_orig: S += 1
                total += 1
                
        if total == 0: return None
            
        diff = round((R / total * 100) - (S / total * 100), 2)
        
        return {
            'payload': None,
            'filename': Path(image_path).name,
            'file_size_kb': round(file_size_kb, 2),
            'diff': diff,
            'status': 'ЧИСТОЕ' if diff > 15 else 'ПОДОЗРИТЕЛЬНО' if diff < 8 else 'СРЕДНЕ'
        }
    except:
        return None


# ===================== АНАЛИЗ =====================
all_results = []
print("Запуск анализа с учётом размера файла...\n")

for payload_name, folder_name in folders.items():
    folder_path = Path(base_folder) / folder_name
    print(f"Обрабатывается: {payload_name}")
    
    paths = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + \
            list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.bmp"))
    paths = paths[:max_images_per_folder]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(rs_analysis_image, p): p for p in paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(paths), desc=payload_name):
            res = future.result()
            if res:
                res['payload'] = payload_name
                results.append(res)
    
    all_results.extend(results)
    print(f"  Готово: {len(results)} изображений\n")

df = pd.DataFrame(all_results)

# ===================== ГРАФИКИ =====================
# 1. Четыре гистограммы diff
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('RS-стегоанализ — Распределение diff (R-S) по объёму внедрения', fontsize=18, fontweight='bold')

colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']

for i, payload in enumerate(folders.keys()):
    ax = axes[i//2, i%2]
    data = df[df['payload'] == payload]['diff']
    ax.hist(data, bins=40, color=colors[i], alpha=0.85, edgecolor='black')
    ax.set_title(f'{payload}', fontsize=14)
    ax.set_xlabel('diff (R - S)')
    ax.set_ylabel('Количество изображений')
    ax.grid(True, alpha=0.3)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее = {data.mean():.2f}')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 2. Связь размера файла и diff
plt.figure(figsize=(11, 8))
plt.scatter(df['file_size_kb'], df['diff'], alpha=0.5, s=20, color='blue')
plt.title('Связь размера файла (КБ) и показателя diff', fontsize=16, pad=15)
plt.xlabel('Размер файла (КБ)', fontsize=12)
plt.ylabel('diff (R - S)', fontsize=12)
plt.grid(True, alpha=0.3)

# Линия тренда
z = np.polyfit(df['file_size_kb'], df['diff'], 1)
p = np.poly1d(z)
plt.plot(df['file_size_kb'], p(df['file_size_kb']), "r--", linewidth=2, label=f'Тренд (наклон = {z[0]:.4f})')
plt.legend()
plt.show()

# ===================== ИТОГИ =====================
print("\n" + "="*90)
print("                          ИТОГОВЫЙ ОТЧЁТ")
print("="*90)

for p in folders.keys():
    data = df[df['payload'] == p]
    high = (data['status'] == 'ПОДОЗРИТЕЛЬНО').sum()
    pct = high / len(data) * 100
    mean_diff = data['diff'].mean()
    print(f"{p:12} | Изобр: {len(data):4} | Подозрительных: {high:4} ({pct:5.1f}%) | Средний diff: {mean_diff:.2f}")

corr = df['file_size_kb'].corr(df['diff'])
print(f"\nКорреляция размер файла ↔ diff: {corr:.3f} "
      f"({'отрицательная' if corr < 0 else 'положительная'})")

print("\nГотово!")