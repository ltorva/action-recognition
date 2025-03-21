import os
from pathlib import Path

def rename_video_files():
    """按顺序重命名视频文件 - 不带后缀"""
    videos_dir = Path('data/videos')
    
    # 确保videos目录存在
    if not videos_dir.exists():
        print(f"错误: {videos_dir} 目录不存在")
        return
    
    total_videos = 0
    renamed_files = 0
    
    # 遍历所有动作类别目录
    for class_dir in sorted(videos_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\n处理类别: {class_name}")
        
        # 获取所有MP4文件
        video_files = sorted(class_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"警告: {class_name} 目录下没有MP4文件")
            continue
            
        print(f"找到 {len(video_files)} 个MP4文件")
        total_videos += len(video_files)
        
        # 遍历并重命名视频文件
        for idx, video_file in enumerate(video_files, 1):
            try:
                # 构建新文件名 (不带后缀)
                new_name = f"{class_name}_{idx:03d}"
                new_path = class_dir / video_file.name
                
                # 如果文件名相同则跳过
                if video_file.stem == new_name:
                    print(f"跳过已重命名的文件: {new_name}")
                    continue
                
                # 构建新路径 (添加原始后缀)
                new_path = class_dir / f"{new_name}{video_file.suffix}"
                
                # 重命名文件
                print(f"重命名: {video_file.name} -> {new_path.name}")
                video_file.rename(new_path)
                renamed_files += 1
                
            except Exception as e:
                print(f"重命名失败 {video_file.name}: {str(e)}")
    
    # 打印统计信息
    print(f"\n重命名完成!")
    print(f"总视频文件数: {total_videos}")
    print(f"成功重命名数: {renamed_files}")

if __name__ == '__main__':
    try:
        rename_video_files()
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())