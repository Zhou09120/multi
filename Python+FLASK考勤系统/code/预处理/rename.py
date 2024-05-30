import os

def rename_images(folder_path, prefix='image', start_index=1):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 仅处理图片文件（根据扩展名过滤）
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # 按顺序重命名
    for index, filename in enumerate(image_files, start=start_index):
        # 获取文件扩展名
        file_extension = os.path.splitext(filename)[1]
        
        # 新文件名
        new_filename = f"{prefix}_{index}{file_extension}"
        
        # 完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

# 示例用法
folder_path = '图片底库\图片底库'  # 替换为你的文件夹路径
rename_images(folder_path)
