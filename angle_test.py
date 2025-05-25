import cv2
import numpy as np
import os

def create_debug_dir():
    debug_dir = "angle_debug"
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
    
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def detect_shapes_with_angle(image_path):
    debug_dir = create_debug_dir()
    
    # 读取图像并预处理
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作
    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        # 先计算中心坐标 --------------------------------------------------
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue  # 跳过无效轮廓
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 多边形近似
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        shape = "unknown"
        angle = 0.0
        
        # ================= 形状识别 & 角度计算 =================
        # 三角形（倒三角形最上边为基准）
        if vertices == 3:
            shape = "triangle"
            
            # 找到最上面的顶点（y坐标最小）
            points = approx.squeeze()
            top_idx = np.argmin(points[:,1])
            top_point = points[top_idx]
            
            # 获取包含顶点的两条边
            edge1 = [top_point, points[(top_idx-1)%3]]
            edge2 = [top_point, points[(top_idx+1)%3]]
            
            base_edge_tri = edge1 
            
            # 计算基准边角度（顺时针为正）
            angle = compute_angle(base_edge_tri[0], base_edge_tri[1]) % 180
            while angle > 120:
                if base_edge_tri == edge1:
                    base_edge_tri = edge2
                else:
                    base_edge_tri = edge1
                angle = compute_angle(base_edge_tri[0], base_edge_tri[1]) % 180

            base_line = [tuple(base_edge_tri[0]), tuple(base_edge_tri[1])]  
            
        # 四边形
        elif vertices == 4:
            # 顶点排序（顺时针）
            points = approx.reshape(-1, 2)  # 变成 (4,2)
            
            # 计算中心点 cx, cy（确保定义）
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])

            # 计算每个点与中心点的角度
            angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
            sort_idx = np.argsort(angles)
            points = points[sort_idx]

            # 计算两条对角线向量
            diag1_vec = points[2] - points[0]
            diag2_vec = points[3] - points[1]
            dot_product = np.dot(diag1_vec, diag2_vec)
            diag1_len = np.linalg.norm(diag1_vec)
            diag2_len = np.linalg.norm(diag2_vec)
            is_diag_ortho = abs(dot_product) < 0.1 * diag1_len * diag2_len  # 判断是否近似垂直

            if is_diag_ortho:
                shape = "diamond" # 以最长对角边为基准
                diag1 = np.linalg.norm(points[2] - points[0])
                diag2 = np.linalg.norm(points[3] - points[1])
                if diag1 > diag2:
                    diag_points = [points[0], points[2]]
                else:
                    diag_points = [points[1], points[3]]
                angle = compute_angle(diag_points[0], diag_points[1]) % 180
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
                
                angle = compute_angle(base_edge_tpd[0], base_edge_tpd[1]) % 180
                base_line = [tuple(base_edge_tpd[0]), tuple(base_edge_tpd[1])]

                # === 寻找可能的“平行短边” ===
                opp_idx = (base_idx + len(points)//2) % len(points)
                opp_pt1 = points[opp_idx]
                opp_pt2 = points[(opp_idx + 1) % len(points)]
                midpoint_base = (base_edge_tpd[0] + base_edge_tpd[1]) / 2
                midpoint_opp = (opp_pt1 + opp_pt2) / 2

                # 判断相对位置
                vec = midpoint_opp - midpoint_base
                if vec[0] < -100 or vec[1] > 100:  # 在左侧或下侧（你可以按需调整）
                    angle = (360 - angle) % 360
            
        # 正六边形（最上边为基准）
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

            # 计算两个边在 x 方向上的分量（水平程度）
            dx1 = pt_prev[0] - top_point[0]
            dx2 = pt_next[0] - top_point[0]

            # 选择更水平的边作为 base
            base_edge_hex = edge1 if dx1 > dx2 else edge2

            angle = compute_angle(base_edge_hex[0], base_edge_hex[1]) % 180
            base_line = [tuple(base_edge_hex[0]), tuple(base_edge_hex[1])]
        
        # ================= 结果保存 =================
        results.append({
            "shape": shape,
            "center": (int(cx), int(cy)),
            "angle": round(angle, 1),
            "contour": cnt,
            "base_line": base_line
        })
    
    # 可视化结果
    output = image.copy()
    for res in results:
        cv2.drawContours(output, [res["contour"]], -1, (0,255,0), 2)
        cv2.circle(output, res["center"], 5, (0,0,255), -1)
        
        # 绘制角度标注
        text = f"{res['shape']} {res['angle']}"
        cv2.putText(output, text, 
                   (res["center"][0]-40, res["center"][1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        
        # 绘制参考线（红色为基准线）
        if "base_line" in res:
            pt1, pt2 = res["base_line"]
            cv2.line(output, pt1, pt2, (0, 0, 255), 2)
    
    save_image(output, "final_result", debug_dir)
    
    # 打印结果
    print("检测结果：")
    for res in results:
        print(f"{res['shape']}: 中心({res['center']}), 偏转角{res['angle']}°")
    
    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = r'imgs\angle_test2.png'
# 测试
detect_shapes_with_angle(img_path)