import os
from pathlib import Path

def rename_videos():
    """重命名子文件夹下的视频文件"""
    videos_dir = Path('data/videos')
    
    # 确保videos目录存在
    if not videos_dir.exists():
        print(f"错误: {videos_dir} 目录不存在")
        return
    
    total_folders = 0
    renamed_files = 0
    
    # 遍历所有动作类别目录
    for class_dir in videos_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\n处理类别: {class_name}")
        
        # 获取所有子文件夹
        subfolders = [d for d in class_dir.iterdir() if d.is_dir()]
        subfolders.sort()  # 按名称排序
        
        if not subfolders:
            print(f"警告: {class_name} 目录下没有子文件夹")
            continue
            
        print(f"找到 {len(subfolders)} 个子文件夹")
        total_folders += len(subfolders)
        
        # 遍历子文件夹
        for idx, subfolder in enumerate(subfolders, 1):
            try:
                # 查找子文件夹中的视频文件
                videos = list(subfolder.glob('*.mp4')) + \
                        list(subfolder.glob('*.avi')) + \
                        list(subfolder.glob('*.mov'))
                
                if not videos:
                    print(f"警告: {subfolder.name} 中没有视频文件")
                    continue
                
                video_file = videos[0]  # 获取第一个视频文件
                new_name = f"{class_name}_{idx:03d}{video_file.suffix.lower()}"
                new_path = class_dir / new_name
                
                # 如果目标文件已存在，添加后缀
                counter = 1
                while new_path.exists():
                    new_name = f"{class_name}_{idx:03d}_{counter}{video_file.suffix.lower()}"
                    new_path = class_dir / new_name
                    counter += 1
                
                try:
                    print(f"正在重命名: {video_file.name} -> {new_name}")
                    video_file.rename(new_path)
                    renamed_files += 1
                except Exception as e:
                    print(f"重命名失败 {video_file.name}: {str(e)}")
            except Exception as e:
                print(f"处理子文件夹 {subfolder.name} 时发生错误: {str(e)}")

    # 打印统计信息
    print(f"\n重命名完成!")
    print(f"总文件夹数: {total_folders}")
    print(f"成功重命名: {renamed_files}")

if __name__ == '__main__':
    try:
        rename_videos()
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())