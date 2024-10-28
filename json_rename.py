import os

def rename_json_to_image_names(img_path, json_path):
    # 이미지 파일과 JSON 파일을 순서대로 가져옵니다.
    img_list = sorted([file for file in os.listdir(img_path) if file.endswith('.png')])
    json_list = sorted([file for file in os.listdir(json_path) if file.endswith('.json')])

    # 이미지 파일과 JSON 파일의 개수가 같은지 확인합니다.
    if len(img_list) != len(json_list):
        print(f"Error: 이미지 파일({len(img_list)}개)과 JSON 파일({len(json_list)}개)의 개수가 다릅니다.")
        return

    # 파일 이름을 하나씩 변경합니다.
    for img_file, json_file in zip(img_list, json_list):
        img_file_name = os.path.splitext(img_file)[0]  # 이미지 파일 이름에서 확장자 제거
        json_old_path = os.path.join(json_path, json_file)  # 기존 JSON 파일 경로
        json_new_path = os.path.join(json_path, f"{img_file_name}.json")  # 새 JSON 파일 이름 경로
        
        # 파일 이름 변경
        os.rename(json_old_path, json_new_path)
        print(f"Renamed: {json_file} -> {img_file_name}.json")

if __name__ == "__main__":
    img_path = "/home/julio981007/HDD/HDX/selected_img/"
    json_path = "/home/julio981007/HDD/HDX/senario_5_preds/"
    
    rename_json_to_image_names(img_path, json_path)
