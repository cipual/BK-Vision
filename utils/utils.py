from collections import Counter
import socket
import psutil

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
