import os
import random
import shutil

'''
# 원본 이미지와 대상 이미지의 디렉토리를 지정합니다.
src_dir = "images/cifar10-32/train"
dst_dir = "images/cifar10-32(2)/train"

# 각 클래스 디렉토리를 순회합니다.
for class_name in os.listdir(src_dir):
    class_src_dir = os.path.join(src_dir, class_name)
    
    # 해당 클래스 디렉토리 내의 모든 이미지 파일 이름을 가져옵니다.
    all_files = os.listdir(class_src_dir)
    
    # 이미지의 20%를 무작위로 선택합니다.
    num_to_select = int(len(all_files) * 0.20)
    selected_files = random.sample(all_files, num_to_select)
    
    # 선택된 이미지들을 새로운 디렉토리에 복사합니다.
    class_dst_dir = os.path.join(dst_dir, class_name)
    if not os.path.exists(class_dst_dir):  # 해당 클래스의 대상 디렉토리가 없으면 생성합니다.
        os.makedirs(class_dst_dir)
    
    for filename in selected_files:
        src_path = os.path.join(class_src_dir, filename)
        dst_path = os.path.join(class_dst_dir, filename)
        shutil.copy(src_path, dst_path)
'''

# 원본 이미지가 있는 디렉토리와 대상 디렉토리를 지정합니다.
src_dir = "images/FFHQ/dataset"
dst_dir = "images/FFHQ2/img"

# 원본 디렉토리에서 모든 파일의 이름을 가져옵니다.
all_files = os.listdir(src_dir)

# 14,000개의 파일 이름을 무작위로 선택합니다.
selected_files = random.sample(all_files, 7000)

# 선택된 파일들을 새로운 디렉토리로 복사하면서 이름을 변경합니다.
for idx, filename in enumerate(selected_files):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, f"{idx}.png")
    shutil.copy(src_path, dst_path)
