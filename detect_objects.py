import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np

#혹시 필요하실까봐 박스 쳐진 이미지 파일도 생성되게 만들었습니다.
#텍스트 파일 이름은 원본 이미지와 같게 만들어지게 작성 됐고, 이미지 파일 이름은 _bbox가 추가된 이름으로 저장됩니다.
#가중치, 사진이 저장돼있는 이미지 폴더, true와 사람 명 수 이름이 작성될 텍스트 폴더 경로, 박스가 쳐지고 나올 폴더 경로 
weight_path = '/content/drive/MyDrive/test/best.pt' #가중치 경로
img_folder_path = '/content/drive/MyDrive/test/test_light' #테스트할 사진 경로
txt_folder_path = '/content/drive/MyDrive/test/light_txt1' #텍스트가 저장되는 경로 
bbox_img_folder_path = '/content/drive/MyDrive/test/light_image' #박스가 쳐진 이미지가 저장되는 경로 


def detect_objects(weight_path, image_folder_path, text_folder_path, bbox_image_folder_path):
    #모델 로드 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)
    #테스트할 이미지 확장자 이름 바꾸시면 됩니다 현재 jpg, png 
    image_files = [f for f in Path(image_folder_path).rglob('*') if f.suffix in ['.jpg', '.png']]
    #텍스트 파일은 첫 행에는 true, false로 저장되고 두 번째 행에는 인원 수가 저장됩니다. 
    for image_file in image_files:
        results = model(image_file)

        object_count = len(results.xyxy[0])
        detection_status = 'true' if object_count > 0 else 'false'
        
        with open(f"{text_folder_path}/{image_file.stem}.txt", "w") as file:
            file.write(f"{detection_status}\n") #첫 행에 true or false 작성
            file.write(f"{object_count}\n") #두 번째 행에 사람 수 작성 

        #이미지 파일 저장
        img_with_boxes_arr = np.array(results.render()[0])
        img_with_boxes = Image.fromarray(img_with_boxes_arr)
        img_with_boxes.save(os.path.join(bbox_image_folder_path, f'{image_file.stem}_bbox.jpg'))

detect_objects(weight_path, img_folder_path, txt_folder_path, bbox_img_folder_path)
