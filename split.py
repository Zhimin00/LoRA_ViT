import os
import shutil
import random

# Set your paths
input_folder = '/cis/home/zshao14/Downloads/RETFound_MAE/CS/raw_test_denoised2'  # Folder with test_0001_AMD.png, etc
output_folder = '/cis/home/zshao14/Downloads/RETFound_MAE/CS/data_denoised'       # Where train/, val/, test/ will be created

# Create temp folders to reorganize by class
temp_folder = os.path.join(output_folder, 'temp')
classes = ['AMD', 'DME']
for cls in classes:
    os.makedirs(os.path.join(temp_folder, cls), exist_ok=True)

# Step 1: Move and rename files into temp/{class}/{index}.png
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
image_files.sort()  # Optional: Sort files first for consistent ordering

class_counters = {cls: 1 for cls in classes}

for img_name in image_files:
    # Example: test_0001_AMD.png
    parts = img_name.split('_')
    if len(parts) < 3:
        print(f"Skipping file {img_name}")
        continue

    class_name_with_ext = parts[2]
    class_name = class_name_with_ext.split('.')[0]  # Get 'AMD' or 'DME'

    if class_name not in classes:
        print(f"Unknown class {class_name} in {img_name}, skipping")
        continue

    # Generate new filename
    idx = class_counters[class_name]
    new_name = f'{idx:04d}.png'  # 0001.png, 0002.png, etc.
    class_counters[class_name] += 1

    src_path = os.path.join(input_folder, img_name)
    dst_path = os.path.join(temp_folder, class_name, new_name)

    shutil.copy(src_path, dst_path)

# Step 2: Split into train/val/test
splits = ['train', 'val', 'test']
split_ratio = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
}

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_folder, split, cls), exist_ok=True)

for cls in classes:
    cls_folder = os.path.join(temp_folder, cls)
    images = [f for f in os.listdir(cls_folder) if f.endswith('.png')]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(split_ratio['train'] * n_total)
    n_val = int(split_ratio['val'] * n_total)
    n_test = n_total - n_train - n_val

    split_counts = {
        'train': n_train,
        'val': n_val,
        'test': n_test,
    }

    start_idx = 0
    for split in splits:
        count = split_counts[split]
        for img_name in images[start_idx:start_idx+count]:
            src_path = os.path.join(cls_folder, img_name)
            dst_path = os.path.join(output_folder, split, cls, img_name)
            shutil.copy(src_path, dst_path)
        start_idx += count

print("Done!")
