import cv2
import numpy as np
import os

print(cv2.__version__)

img_path = r'imgs\shiyan1.png'
txt_path = r'config\waican.txt'
Trans = []
with open(txt_path, 'r') as f:
    for line in f:
        parts = [float(x.strip()) for x in line.strip().split(',') if x.strip()]
        Trans.append(parts)
Trans = np.array(Trans)  # 3x3，列主序
Trans = Trans.T  

def transform(x, y):
    point_h = np.array([x, y, 1])  # 齐次坐标
    transformed = Trans @ point_h
    return transformed[0], transformed[1]

def create_debug_dir():
    debug_dir = "shape_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir

def save_image(image, step_name, debug_dir):
    cv2.imwrite(f"{debug_dir}/{step_name}.png", image)

def compute_angle(pt1, pt2):
    # 确保 pt1 在左，pt2 在右
    if pt1[0] > pt2[0]:  # 强制从左向右
        pt1, pt2 = pt2, pt1
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return angle

def detect_shapes(image_path):
    debug_dir = create_debug_dir()

    image = cv2.imread(image_path)
    save_image(image, "01_original", debug_dir)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    save_image(hsv, "02_hsv_blur", debug_dir)

    color_ranges = {
        "red": ([150, 50, 40], [180, 200, 255]),
        'yellow': [(5, 80, 60), (30, 255, 255)],
        'green' :[(10, 20, 10), (90, 120, 65)],
    }

    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    save_image(combined_mask, "03_color_mask", debug_dir)

    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    save_image(closed, "04_closed", debug_dir)

    filtered_mask = cv2.medianBlur(closed, 7)
    save_image(filtered_mask, "05_filtered_mask", debug_dir)

    edges = cv2.Canny(filtered_mask, 50, 120)
    save_image(edges, "06_edges", debug_dir)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        shape = "unknown"
        angle = 0.0

        # ================= 形状识别 =================
        if vertices == 3:
            shape = "triangle"
            
            # 找到最上面的顶点（y坐标最小）
            points = approx.squeeze()
            top_idx = np.argmin(points[:,1])
            top_point = points[top_idx]
            
            # 获取包含顶点的两条边
            edge1 = [top_point, points[(top_idx - 1) % 3]]
            edge2 = [top_point, points[(top_idx + 1) % 3]]

            # 计算两个边的x方向变化量
            dx1 = edge1[1][0] - edge1[0][0]
            dx2 = edge2[1][0] - edge2[0][0]

            # 选择x方向变化量更大的（更“靠右”）那条边
            base_edge_tri = edge1 if dx1 > dx2 else edge2
        
            # 计算基准边角度（顺时针为正）
            angle = (360 - compute_angle(base_edge_tri[0], base_edge_tri[1])) % 120
            base_line = [tuple(base_edge_tri[0]), tuple(base_edge_tri[1])]

        elif vertices == 4:
            # 判断是菱形还是梯形
            points = approx.reshape(-1, 2)
            cx_poly = np.mean(points[:, 0])
            cy_poly = np.mean(points[:, 1])
            angles_poly = np.arctan2(points[:, 1] - cy_poly, points[:, 0] - cx_poly)
            sort_idx = np.argsort(angles_poly)
            points = points[sort_idx]

            diag1_vec = points[2] - points[0]
            diag2_vec = points[3] - points[1]
            dot_product = np.dot(diag1_vec, diag2_vec)
            diag1_len = np.linalg.norm(diag1_vec)
            diag2_len = np.linalg.norm(diag2_vec)
            is_diag_ortho = abs(dot_product) < 0.1 * diag1_len * diag2_len

            if is_diag_ortho:
                shape = "diamond" # 以最长对角边为基准
                diag1 = np.linalg.norm(points[2] - points[0])
                diag2 = np.linalg.norm(points[3] - points[1])
                if diag1 > diag2:
                    diag_points = [points[0], points[2]]
                else:
                    diag_points = [points[1], points[3]]
                angle = 180 - compute_angle(diag_points[0], diag_points[1])
                base_line = [tuple(diag_points[0]), tuple(diag_points[1])]
            else:
                shape = "trapezoid"
                max_len = 0
                base_edge_tpd = None
                base_idx = 0
                
                for i in range(len(points)):
                    pt1 = points[i]
                    pt2 = points[(i + 1) % len(points)]  # 邻接点构成边
                    length = np.linalg.norm(pt1 - pt2)
                    if length > max_len:
                        max_len = length
                        base_edge_tpd = [pt1, pt2]
                        base_idx = i
                
                angle = 360 - compute_angle(base_edge_tpd[0], base_edge_tpd[1])
                base_line = [tuple(base_edge_tpd[0]), tuple(base_edge_tpd[1])]

                # === 寻找可能的“平行短边” ===
                # 与基准边对面的边索引，假设图形轮廓点有4~6个时成立
                opp_idx = (base_idx + len(points)//2) % len(points)
                opp_pt1 = points[opp_idx]
                opp_pt2 = points[(opp_idx + 1) % len(points)]
                midpoint_base = (base_edge_tpd[0] + base_edge_tpd[1]) / 2
                midpoint_opp = (opp_pt1 + opp_pt2) / 2

                # 判断相对位置
                vec = midpoint_opp - midpoint_base
                if vec[0] < -50 or vec[1] > 50:  
                    angle = (360 - angle) % 360

        elif vertices == 6:
            shape = "hexagon"
            
            points = approx.squeeze()  # shape (6, 2)
            # 找到最上方的顶点（y 最小）
            top_idx = np.argmin(points[:, 1])
            top_point = points[top_idx]

            # 获取与顶点相邻的两条边
            prev_idx = (top_idx - 1) % 6
            next_idx = (top_idx + 1) % 6
            pt_prev = points[prev_idx]
            pt_next = points[next_idx]

            edge1 = [top_point, pt_prev]
            edge2 = [top_point, pt_next]

            # 计算两个边在 x 方向上的分量
            dx1 = pt_prev[0] - top_point[0]
            dx2 = pt_next[0] - top_point[0]

            # 选择更靠右的边作为 baseline
            base_edge_hex = edge1 if dx1 > dx2 else edge2

            angle = (360 - compute_angle(base_edge_hex[0], base_edge_hex[1])) % 60
            base_line = [tuple(base_edge_hex[0]), tuple(base_edge_hex[1])]

        if shape != 'unknown':
            print(transform(cx, cy))
            results.append({
                "shape": shape,
                "center": (int(cx), int(cy)),
                "angle": round(angle, 1),
                "contour": cnt,
                "base_line": base_line,
            })

    output = image.copy()
    for res in results:
        cv2.drawContours(output, [res["contour"]], -1, (0,255,0), 2)
        cv2.circle(output, res["center"], 3, (0,0,255), -1)
        text = f"{res['shape']} {res['angle']}"
        cv2.putText(output, text, 
                   (res["center"][0]-40, res["center"][1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if "base_line" in res:
            pt1, pt2 = res["base_line"]
            cv2.line(output, pt1, pt2, (0, 0, 255), 2)


    save_image(output, "final_result", debug_dir)

    print("检测结果：")
    for res in results:
        print(f"{res['shape']}: 中心({res['center']}), 偏转角{res['angle']}°")

    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 测试
detect_shapes(img_path)