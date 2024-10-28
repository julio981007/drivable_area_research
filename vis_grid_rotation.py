import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.colors as mcolors
from matplotlib.path import Path
from collections import deque


# Flood-fill 알고리즘으로 바운딩 박스를 채우는 함수
def flood_fill(gridmap, start_x, start_y):
    fill_value = 1  # 빨간색으로 채우기
    rows, cols = gridmap.shape

    if gridmap[start_y, start_x] == fill_value:
        return gridmap

    # 너비 우선 탐색을 사용한 flood-fill
    queue = deque([(start_x, start_y)])
    while queue:
        x, y = queue.popleft()

        if x < 0 or x >= cols or y < 0 or y >= rows:
            continue

        if gridmap[y, x] == fill_value:
            continue

        # 현재 위치 채우기
        gridmap[y, x] = fill_value

        # 상하좌우로 확장
        queue.append((x + 1, y))
        queue.append((x - 1, y))
        queue.append((x, y + 1))
        queue.append((x, y - 1))

    return gridmap

# 2D 회전을 적용하는 함수
def apply_rotation(x, y, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_new = cos_theta * x - sin_theta * y
    y_new = sin_theta * x + cos_theta * y
    return x_new, y_new

    
# 바운딩 박스 시각화 함수
def visualize_bounding_boxes_on_gridmap(gridmap, bounding_boxes, grid_size=50, resolution=0.5):
    for bbox in bounding_boxes:
        # 바운딩 박스 중심 좌표
        x_center, y_center, z_center = bbox[:3]
        length, width, height = bbox[3:6]
        rotation_z = bbox[6]  # Z축 회전 정보 (radians)

        # 바운딩 박스의 4개 꼭짓점 좌표 계산
        corners = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2]
        ])

        # 각 꼭짓점에 회전 행렬 적용
        rotated_corners = []
        for corner in corners:
            rotated_corner = apply_rotation(corner[0], corner[1], rotation_z)
            rotated_corners.append(rotated_corner)

        rotated_corners = np.array(rotated_corners)

        # 꼭짓점 좌표를 중심 좌표에 맞게 이동
        rotated_corners[:, 0] += x_center + grid_size / 2
        rotated_corners[:, 1] += y_center + grid_size / 2

        # 그리드 좌표로 변환
        grid_x = (rotated_corners[:, 0] / resolution).astype(int)
        grid_y = (rotated_corners[:, 1] / resolution).astype(int)

        # 그리드 좌표를 그리드 맵의 범위 내로 클리핑
        grid_x = np.clip(grid_x, 0, gridmap.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, gridmap.shape[0] - 1)
        

        # 바운딩 박스를 그리드에 채우기
        for i in range(len(grid_x)):
            next_i = (i + 1) % len(grid_x)
            rr = np.linspace(grid_y[i], grid_y[next_i], num=100).astype(int)
            cc = np.linspace(grid_x[i], grid_x[next_i], num=100).astype(int)
            
            # rr과 cc의 범위도 그리드 크기 내로 클리핑
            rr = np.clip(rr, 0, gridmap.shape[0] - 1)
            cc = np.clip(cc, 0, gridmap.shape[1] - 1)
            
            gridmap[rr, cc] = 1  # 빨간색 바운딩 박스 표시

        # 바운딩 박스 중심을 그리드 좌표로 변환
        grid_x_center = int((x_center + grid_size / 2) / resolution)
        grid_y_center = int((y_center + grid_size / 2) / resolution)

        # 중심에서 flood-fill을 이용해 바운딩 박스 내부 채우기
        gridmap = flood_fill(gridmap, grid_x_center, grid_y_center)



    # 중심 (0,0)에서부터 가로 세로 5미터 범위를 검정색으로 덧칠하기
    grid_x_min = int((-5 + grid_size / 2) / resolution)
    grid_x_max = int((5 + grid_size / 2) / resolution)
    grid_y_min = int((-5 + grid_size / 2) / resolution)
    grid_y_max = int((5 + grid_size / 2) / resolution)

    for grid_x in range(max(0, grid_x_min), min(gridmap.shape[1], grid_x_max + 1)):
        for grid_y in range(max(0, grid_y_min), min(gridmap.shape[0], grid_y_max + 1)):
            gridmap[grid_y, grid_x] = 0  # 검정색으로 덧칠 (0은 검정색)

    # 중심 (0,0)을 연두색으로 칠하기 (연두색 값은 2로 설정)
    center_x = int(grid_size / 2 / resolution)
    center_y = int(grid_size / 2 / resolution)

    # (-2, -1) ~ (2, 1) 영역을 연두색으로 칠하기
    grid_x_min = center_x - int(2 / resolution) - 1
    grid_x_max = center_x + int(2 / resolution)
    grid_y_min = center_y - int(1 / resolution) - 1
    grid_y_max = center_y + int(1 / resolution)

    # 해당 범위를 연두색(값 2)으로 채우기
    for grid_x in range(grid_x_min, grid_x_max + 1):
        for grid_y in range(grid_y_min, grid_y_max + 1):
            if 0 <= grid_x < gridmap.shape[1] and 0 <= grid_y < gridmap.shape[0]:
                gridmap[grid_y, grid_x] = 2  # 연두색으로 설정
    
    return gridmap

# 바운딩 박스 로드
def load_bounding_boxes(json_path, score_threshold=0.2):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # scores_3d가 0.2 이상인 객체들만 필터링
    filtered_bboxes = []
    for score, bbox in zip(data['scores_3d'], data['bboxes_3d']):
        if score >= score_threshold:
            filtered_bboxes.append(bbox)
    
    return filtered_bboxes

# 바운딩 박스만 시각화
def create_gridmap_and_visualize(json_path, grid_size=50, resolution=0.5):
    # 그리드맵 초기화 (빈 그리드맵 생성)
    gridmap = np.zeros((int(grid_size / resolution), int(grid_size / resolution)), dtype=np.uint8)

    # 바운딩 박스 로드 (점수 0.2 이상인 객체들만 필터링)
    bounding_boxes = load_bounding_boxes(json_path, score_threshold=0.2)
    
    # 바운딩 박스 추가한 그리드맵 생성
    gridmap_with_bboxes = visualize_bounding_boxes_on_gridmap(gridmap, bounding_boxes, grid_size, resolution)
    
    # 컬러맵 설정 (검정색, 빨간색, 연두색)
    cmap = mcolors.ListedColormap(['black', 'red', 'Lime'])  
    bounds = [0, 1, 2, 3]  # 세 가지 색상을 나타내기 위한 경계값 설정
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 그리드맵 시각화
    gridmap_with_bboxes = np.flipud(gridmap_with_bboxes)
    gridmap_with_bboxes = np.rot90(gridmap_with_bboxes)
    plt.imshow(gridmap_with_bboxes, cmap=cmap, norm=norm, extent=[-25, 25, 0, 50])
    
    # 그리드 선을 resolution 값에 맞게 표시
    plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
    plt.gca().set_yticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
    
    # 그리드 선 표시 (세부적인 그리드 표시)
    plt.gca().grid(False, which='minor', color='gray', linestyle='-', linewidth=0.5)  # 세부 그리드 표시

    # 주 그리드 선도 5미터 단위로 표시
    plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
    plt.gca().set_yticks(np.arange(-grid_size/2, grid_size/2 + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)

    plt.gca().grid(True, which='major', color='gray', linestyle='-', linewidth=1.0)
    # plt.ylim(0,25)
    plt.title('Gridmap with Bounding Boxes (Ego = Green, Object = Red)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # 한 번에 플롯을 그려서 한 창에서 표시
    plt.show()

# 실행 코드
json_file = "/mnt/HDD/HDX/senario_5_preds/1728352330.176256418.json"

create_gridmap_and_visualize(json_file)