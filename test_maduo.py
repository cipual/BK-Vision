import cv2
import numpy as np
import os

# 加载图像
img_path = r'imgs\maduo_test1.png'

def create_debug_dir():
    """创建调试目录"""
    debug_dir = "debug_results"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir

def save_image(image, step_name, debug_dir):
    """保存调试图像"""
    cv2.imwrite(f"{debug_dir}/{step_name}.png", image)

def detect_shapes_and_colors(image_path):
    # 创建调试目录
    debug_dir = create_debug_dir()
    
    # 读取原始图像
    image = cv2.imread(image_path)
    save_image(image, "01_original", debug_dir)
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    save_image(hsv, "02_hsv_blur", debug_dir)
    
    # 颜色阈值定义（根据实际调整）
    color_ranges = {
        "blue": ([90, 85, 13], [160, 255, 255]),   # 调整蓝色范围
        "red": ([0, 100, 20], [100, 255, 255]),     # 红色低区间
        'yellow': [(10, 60, 13), (30, 255, 255)],  # 精确黄色范围
    }
    
    # 生成颜色掩膜
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    save_image(combined_mask, "03_color_mask", debug_dir)
    
    # 形态学操作去除小孔洞
    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    save_image(closed_mask, "04_closed_mask", debug_dir)
    
    # 填充孔洞
    mask_inv = cv2.bitwise_not(closed_mask)
    cv2.floodFill(mask_inv, None, (0, 0), 255)
    mask_filled = cv2.bitwise_not(mask_inv)
    save_image(mask_filled, "05_filled_mask", debug_dir)
    
    # 中值滤波去噪
    filtered_mask = cv2.medianBlur(mask_filled, 7)
    save_image(filtered_mask, "06_filtered_mask", debug_dir)
    
    # 边缘检测
    edges = cv2.Canny(filtered_mask, 50, 120)
    save_image(edges, "07_edges", debug_dir)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 忽略小面积噪声
            continue
        
        # --- 形状检测 ---
        shape = "unknown"
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # 圆形检测条件：面积与周长比接近圆形
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # 正方形检测条件：多边形近似+宽高比
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if 0.9 < aspect_ratio < 1.1:  # 宽高比接近1
                shape = "square"
        elif circularity > 0.85:  # 圆形判定
            shape = "circle"
        
        # --- 颜色检测 ---
        # 取中心点周围5x5区域的平均颜色
        mask_roi = np.zeros_like(filtered_mask)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        hsv_roi = cv2.bitwise_and(hsv, hsv, mask=mask_roi)
        mean_hue = np.mean(hsv_roi[:,:,0][mask_roi == 255])
        
        color = "unknown"
        if 90 <= mean_hue <= 160:
            color = "blue"
        elif (0 <= mean_hue <= 10) or (170 <= mean_hue <= 180):  # 红色在HSV环的两端
            color = "red"
        elif 10 <= mean_hue <= 30:
            color = "yellow"
        
        # 保存结果
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            results.append({
                "shape": shape,
                "color": color,
                "center": (cx, cy),
                "contour": cnt
            })
    
    # 可视化结果
    output = image.copy()
    for res in results:
        # 绘制轮廓和中心
        cv2.drawContours(output, [res["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(output, res["center"], 5, (0, 0, 255), -1)
        cv2.putText(output, f"{res['shape']}-{res['color']}", 
                    (res["center"][0]-20, res["center"][1]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    save_image(output, "08_final_result", debug_dir)
    
    # 打印结果
    print("检测结果：")
    for res in results:
        print(f"形状: {res['shape']}, 颜色: {res['color']}, 中心坐标: {res['center']}")
    
    cv2.imshow("Final Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 测试
detect_shapes_and_colors(img_path)