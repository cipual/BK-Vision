import matplotlib.pyplot as plt
import numpy as np

# 中文字体与负号设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 定义图形点与属性（位置, 类型, 角度）
shapes = [
    {"pos": (-23.0, -17.7), "type": "triangle", "angle": 60},
    {"pos": (69.0, -18.4), "type": "triangle", "angle": 60},
    {"pos": (-9.2, 27.8), "type": "triangle", "angle": 0},
    {"pos": (52.4, 27.1), "type": "triangle", "angle": 0},
    {"pos": (21.7, 81.5), "type": "triangle", "angle": 0},
    {"pos": (45.1, -22.7), "type": "diamond", "angle": 150},
    {"pos": (-0.5, -22.4), "type": "diamond", "angle": 30},
    {"pos": (61.2, 4.6), "type": "diamond", "angle": 150},
    {"pos": (-16.3, 4.6), "type": "diamond", "angle": 30},
    {"pos": (21.8, 45.2), "type": "hexagon", "angle": 0},
    {"pos": (23.0, 4.3), "type": "trapezoid", "angle": 180},
]

# 统一边长
GRID_SIZE = 15  # 调整此值以匹配原始图形大小

# 绘图初始化
fig, ax = plt.subplots(figsize=(10, 8))

# 隐藏左边和底部的坐标轴边框
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')

# 将刻度标签移到顶部和右侧
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('right')

# 反转坐标轴方向（X向左为正，Y向下为正）
ax.invert_xaxis()
ax.invert_yaxis()

# 绘制函数：根据类型绘制形状
def draw_shape(ax, pos, shape_type, angle_deg, size=GRID_SIZE):
    x0, y0 = pos
    angle_rad = np.radians(angle_deg)
    shape = []

    if shape_type == "triangle":
        h = size * 2 * (3 ** 0.5) / 2
        shape = np.array([[0, 0], [size * 2, 0], [size, h]])
    elif shape_type == "diamond":
        h = size * 2 * (3 ** 0.5) / 2
        shape = np.array([[h, 0], [2*h, size], [h, size * 2], [0, size]])
    elif shape_type == "trapezoid":
        h = size * 2 * (3 ** 0.5) / 2
        shape = np.array([[0, 0], [size * 2, 0], [size*3, h], [-size, h]])
    elif shape_type == "hexagon":
        shape = np.array([
            [size, 0],
            [3*size, 0],
            [4*size, size * (3 ** 0.5)],
            [3*size, 2*size * (3 ** 0.5)],
            [size, 2*size * (3 ** 0.5)],
            [0, size * (3 ** 0.5)]
        ])
    else:
        return

    # 计算几何中心
    center = np.mean(shape, axis=0)
    
    # 将坐标转换为相对于几何中心
    shape_centered = shape - center
    
    # 使用标准旋转矩阵
    rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 应用旋转
    shape_rotated = shape_centered @ rot.T
    
    # 转换到全局坐标系（关键修改：直接平移到目标位置）
    shape_final = shape_rotated + np.array([x0, y0])
    
    polygon = plt.Polygon(shape_final, edgecolor='black', facecolor='skyblue', linewidth=1.2)
    ax.add_patch(polygon)

# 绘制所有图形
for shape in shapes:
    draw_shape(ax, shape["pos"], shape["type"], shape["angle"])

# 设置坐标范围
ax.set_xlim(-50, 100)
ax.set_ylim(-50, 100)

# 添加坐标轴箭头（使用相对坐标定位）
ax.annotate('', xy=(0.05, 1), xycoords='axes fraction',  # X轴箭头（向左）
            xytext=(0.95, 1), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(1, 0.05), xycoords='axes fraction',  # Y轴箭头（向下）
            xytext=(1, 0.95), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 标题与标签
ax.set_title('图形分布图（右上角为原点）', pad=20, fontsize=15)
ax.set_xlabel('X轴（水平向左为正）', fontsize=12, labelpad=30)
ax.set_ylabel('Y轴（垂直向下为正）', fontsize=12, labelpad=30)

# 设置刻度
ax.set_xticks(np.arange(-50, 101, 10))
ax.set_yticks(np.arange(-50, 101, 10))
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()