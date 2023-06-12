import os
import shutil

source_dir = "/home/work/LSGAN_CelebA/LSGAN_CelebA-main/images/백내장"  # 백내장 디렉토리 경로
target_dir = "/home/work/LSGAN_CelebA/LSGAN_CelebA-main/images/CPS_Dog_Cataract"  # 새로운 디렉토리 경로

# 서브 디렉토리 목록
sub_dirs = ["무", "비성숙", "성숙", "초기"]

# 타깃 디렉토리 생성
os.makedirs(target_dir, exist_ok=True)

# 복사된 파일 수를 세기 위한 변수
total_copied_files = 0

# 각 서브 디렉토리에서 이미지 파일 스크랩
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(source_dir, sub_dir)

    # 디렉토리 내의 파일 목록 가져오기
    file_list = os.listdir(sub_dir_path)

    # 이미지 파일만 선택하여 타깃 디렉토리에 복사
    for file_name in file_list:
        if file_name.endswith(".jpg"):
            # 이미지 파일의 경로
            file_path = os.path.join(sub_dir_path, file_name)

            # 이미지 파일을 타깃 디렉토리로 복사
            shutil.copy(file_path, target_dir)

            # 복사된 파일 수 증가
            total_copied_files += 1

print(f"이미지 파일 스크랩이 완료되었습니다. 복사된 파일 수: {total_copied_files}")

#%%
import os

directory = "images/CPS_Dog_Cataract_Rand"  # 디렉토리 경로
image_extensions = [".jpg", ".jpeg", ".png", ".gif"]  # 이미지 파일 확장자

# 디렉토리 내의 파일 목록 가져오기
file_list = os.listdir(directory)

# 이미지 파일 개수를 세는 변수
image_count = 0

# 이미지 파일인지 확인하고 개수 세기
for file_name in file_list:
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension in image_extensions:
        image_count += 1

print(f"디렉토리 내 이미지 파일 개수: {image_count}")
# %%
import os

directory = "images/CPS_Dog_Cataract_Rand"  # 디렉토리 경로
image_extensions = [".jpg", ".jpeg", ".png", ".gif"]  # 이미지 파일 확장자

# 디렉토리 내의 파일 목록 가져오기
file_list = os.listdir(directory)

# 이미지 파일 개수를 세는 변수
image_count = 0

# 이미지 파일의 이름 변경
for file_name in file_list:
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension in image_extensions:
        # 이미지 파일 경로
        file_path = os.path.join(directory, file_name)

        # 새로운 이미지 파일 이름 생성
        new_file_name = "image" + str(image_count) + file_extension

        # 새로운 이름으로 이미지 파일 변경
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(file_path, new_file_path)

        image_count += 1

print("이미지 파일 이름 변경이 완료되었습니다.")
# %%
import os
import random
import shutil

source_dir = "images/CPS_Dog_Cataract"  # 원본 디렉토리 경로
target_dir = "images/CPS_Dog_Cataract_Rand"  # 대상 디렉토리 경로
num_images = 8192  # 추출할 이미지 파일 개수

# 원본 디렉토리 내의 이미지 파일 목록 가져오기
file_list = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            file_list.append(os.path.join(root, file))

# 이미지 파일 목록에서 랜덤하게 num_images 개수만큼 추출
random_images = random.sample(file_list, num_images)

# 대상 디렉토리 생성
os.makedirs(target_dir, exist_ok=True)

# 추출한 이미지 파일들을 대상 디렉토리로 옮기기
for image_path in random_images:
    image_file = os.path.basename(image_path)
    target_path = os.path.join(target_dir, image_file)
    shutil.copy(image_path, target_path)

print(f"{num_images}개의 이미지 파일이 성공적으로 추출되어 {target_dir}로 이동되었습니다.")

directory = "images/CPS_Dog_Cataract_Rand"  # 디렉토리 경로
image_extensions = [".jpg", ".jpeg", ".png", ".gif"]  # 이미지 파일 확장자

# 디렉토리 내의 파일 목록 가져오기
file_list = os.listdir(directory)

# 이미지 파일 개수를 세는 변수
image_count = 0

# 이미지 파일의 이름 변경
for file_name in file_list:
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension in image_extensions:
        # 이미지 파일 경로
        file_path = os.path.join(directory, file_name)

        # 새로운 이미지 파일 이름 생성
        new_file_name = "image" + str(image_count) + file_extension

        # 새로운 이름으로 이미지 파일 변경
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(file_path, new_file_path)

        image_count += 1

print("이미지 파일 이름 변경이 완료되었습니다.")
# %%
