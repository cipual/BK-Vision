from collections import Counter
import socket
import psutil
import cv2
import numpy as np

def count_shapes(results):
    shape_counter = Counter()

    for item in results:
        shape = item.get("shape")
        if shape:
            shape_counter[shape] += 1

    return dict(shape_counter)

def match(results, shapes_data):
    # 定义匹配顺序优先级
    shape_priority = {"triangle": 0, "diamond": 1, "hexagon": 2, "trapezoid": 3}

    # 根据优先级对 results 排序
    results_sorted = sorted(results, key=lambda x: shape_priority.get(x["shape"], 999))

    matched_list = []
    used_indices = set()

    for result in results_sorted:
        r_shape = result["shape"]

        matched_shape = None
        for idx, shape_item in enumerate(shapes_data):
            if idx in used_indices:
                continue
            if shape_item["shape"] == r_shape:
                matched_shape = shape_item
                used_indices.add(idx)
                break

        matched_list.append({
            "result": result,
            "canvas": matched_shape  # 如果找不到，默认就是 None
        })

    return matched_list

def match_one(result, shapes_data, used_indices):
    """
    匹配一个 result 到 shapes_data 中未被使用的图形。
    匹配成功后自动标记为已用。
    """
    r_shape = result["shape"]

    for idx, shape_item in enumerate(shapes_data):
        if idx in used_indices:
            continue
        if shape_item["shape"] == r_shape:
            used_indices.add(idx)  
            return {
                "result": result,
                "canvas": shape_item
            }
    return {
        "result": result,
        "canvas": None
    }

def generate_vision_string(shape_counter):
    # 获取每种图形的数量（没有的默认为 0）
    triangle = shape_counter.get("triangle", 0)
    diamond = shape_counter.get("diamond", 0)
    hexagon = shape_counter.get("hexagon", 0)
    trapezoid = shape_counter.get("trapezoid", 0)

    # 按照通信协议格式拼接字符串
    result_str = f"00OK,S{triangle},P{diamond},L{hexagon},T{trapezoid}*"
    return result_str

def get_local_ips():
    ip_list = set()

    # 使用 psutil 获取所有网卡的IP
    for iface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == socket.AF_INET and not snic.address.startswith("127."):
                ip_list.add(snic.address)

    return sorted(ip_list) if ip_list else ["127.0.0.1"]

def detect_and_segment(frame, predictor, yolo_model):
    """
    使用YOLO检测图像中的目标，并使用SAM对检测到的目标进行分割。
    仅处理类别为“trapezoid”的目标。
    """
    H, W = frame.shape[:2]

    results = yolo_model(frame, imgsz=512)
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for result in results:
        if result.masks is not None and len(result.masks.data) > 0:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            # 获取类别索引和标签名称
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N] 每个框的类别ID
            class_names = [yolo_model.names[cid] for cid in class_ids]  # [N] 对应的类别名

            for i, (cid, cname) in enumerate(zip(class_ids, class_names)):
                if cname != 'trapezoid': 
                    continue
                # 预处理mask和框
                mask = (masks[i] * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

                x1, y1, x2, y2 = boxes[i].astype(int)
                roi_image = frame[y1:y2, x1:x2]
                roi_mask = mask_bin[y1:y2, x1:x2]

                # 计算掩码质心作为SAM提示点
                M = cv2.moments(roi_mask)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = (x2 - x1) // 2
                    cy = (y2 - y1) // 2
                    print(f"[警告] 第{i}个目标掩码为空，使用中心点代替")

                predictor.set_image(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
                input_point = np.array([[cx, cy]])
                input_label = np.array([1])
                masks_sam, _, _ = predictor.predict(point_coords=input_point,
                                                    point_labels=input_label,
                                                    multimask_output=False)

                # 若 SAM 返回空掩码，跳过
                if masks_sam[0].sum() == 0:
                    print(f"[跳过] SAM未成功分割第{i}个目标，point: {cx},{cy}")
                    continue

                # SAM掩码写入原图对应位置
                sam_mask = masks_sam[0].astype(np.uint8) * 255
                sam_mask_full = np.zeros((H, W), dtype=np.uint8)
                sam_mask_full[y1:y2, x1:x2] = sam_mask

                # 融合
                final_mask = cv2.bitwise_or(final_mask, sam_mask_full)
    return final_mask

if __name__ == "__main__":

    shapes_data = [{'shape': 'triangle', 'center': (-23.0, -17.7), 'angle': 60}, 
                   {'shape': 'triangle', 'center': (69.0, -18.4), 'angle': 60}, 
                   {'shape': 'triangle', 'center': (-9.2, 27.8), 'angle': 0}, 
                   {'shape': 'triangle', 'center': (52.4, 27.1), 'angle': 0}, 
                   {'shape': 'triangle', 'center': (21.7, 81.5), 'angle': 0}, 
                   {'shape': 'diamond', 'center': (45.1, -22.7), 'angle': 150}, 
                   {'shape': 'diamond', 'center': (-0.5, -22.4), 'angle': 30}, 
                   {'shape': 'diamond', 'center': (61.2, 4.6), 'angle': 150}, 
                   {'shape': 'diamond', 'center': (-16.3, 4.6), 'angle': 30}, 
                   {'shape': 'hexagon', 'center': (21.8, 45.2), 'angle': 0}, 
                   {'shape': 'trapezoid', 'center': (23.0, 4.3), 'angle': 180
                    }]

    results = [{'shape': 'triangle', 'center': (8.5343, 48.8262), 'angle': 79.9}, 
               {'shape': 'diamond', 'center': (-37.7823, 45.3382), 'angle': 32.7}, 
               {'shape': 'triangle', 'center': (36.9496, 33.1617), 'angle': 79.9}, 
               {'shape': 'diamond', 'center': (-46.2032, 4.204), 'angle': 113.5}, 
               {'shape': 'hexagon', 'center': (-1.0305, 3.1684), 'angle': 17.2}, 
               {'shape': 'trapezoid', 'center': (58.9223, -3.6887), 'angle': 267.3}, 
               {'shape': 'triangle', 'center': (55.6066, -39.7812), 'angle': 28.5}, 
               {'shape': 'diamond', 'center': (21.3142, -39.7056), 'angle': 35.7}, 
               {'shape': 'diamond', 'center': (-49.5856, -34.6145), 'angle': 118.7}, 
               {'shape': 'triangle', 'center': (-13.9804, -42.9335), 'angle': 82.9}]
    matched_list = match(results, shapes_data)
    print(count_shapes(results=results))
    print(generate_vision_string(count_shapes(results=results)))
    print(matched_list)
