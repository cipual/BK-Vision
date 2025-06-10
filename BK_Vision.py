import sys
import json
import threading
import cv2
import numpy as np
import socket
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPolygonItem, QLineEdit, QGridLayout, QGroupBox, QTextEdit, QDesktopWidget, 
    QSizePolicy, QSplitter, QSpinBox, QComboBox, QFileDialog
)
from PyQt5.QtGui import QPolygonF, QBrush, QColor, QPen, QPixmap, QImage, QFont, QTransform
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, pyqtSignal, QThread
from driver.HKcamera import Camera
from utils.utils import match, count_shapes, generate_vision_string, match_one, get_local_ips, detect_and_segment
from time import sleep
import time
sam_path = r"..\..\segment-anything\ "
if sam_path not in sys.path:
    sys.path.append(sam_path)
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
QApplication.setFont(QFont("Microsoft YaHei", 12))

# ------------------- 工具类 -----------------------
class ImgDetector:
    def __init__(self, img_path=r'imgs\20250527_115237.jpg', txt_path=r'config\waican.txt',
                 yolo_model_path = r"weights\best.pt",
                 sam_checkpoint=r"weights\sam_vit_b_01ec64.pth", sam_type="vit_b", device="cuda:0"):
        self.img_path = img_path
        self.Trans = np.round(np.loadtxt(txt_path), 6)
        # 加载模型
        self.yolo_model_path = yolo_model_path
        self.sam_type = sam_type
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.yolo_model = None
        self.sam_predictor = None
        self.start_model_loading_thread()

    def load_models(self):
        from ultralytics import YOLO
        from segment_anything import sam_model_registry, SamPredictor

        # 加载模型（耗时操作）
        yolo_model = YOLO(self.yolo_model_path)
        sam = sam_model_registry[self.sam_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        sam_predictor = SamPredictor(sam)

        # 加载完成后，回到主线程赋值（不能操作 UI，但可以发信号或者设置模型对象）
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        print("模型加载完成")

    def start_model_loading_thread(self):
        thread = threading.Thread(target=self.load_models)
        thread.start()
    
    def compute_angle(self, shape, pt1, pt2):
        if pt1[0] > pt2[0]:
            pt1, pt2 = pt2, pt1
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        if shape == 'hexagon' or shape == 'triangle':
            return np.degrees(np.arctan2(dy, dx)) % 180
        else:
            return np.degrees(np.arctan2(-dy, dx)) % 360  
    
    def transform(self, x, y):
        point_h = np.array([x, y, 1])  # 齐次坐标
        transformed = self.Trans @ point_h
        return transformed[0], transformed[1]
    
    def enhance_contrast_clahe(self, image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


    def enhance_green_saturation_and_light(self, image_bgr, brightness=0.55, contrast=0.9, sat_factor=1.5):
        # 调整对比度和亮度 ---
        image_bgr = cv2.convertScaleAbs(image_bgr, alpha=contrast, beta=0)  # 对比度
        image_bgr = (image_bgr * brightness).clip(0, 255).astype(np.uint8)  # 亮度

        # 转换 HSV 增强绿色饱和度 ---
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 创建绿色区域掩码（Hue 在 [35, 85] 之间）
        green_mask = cv2.inRange(h, 35, 85)
        
        # 饱和度增强
        s_boost = s.astype(np.float32)
        s_boost[green_mask > 0] *= sat_factor
        s_boost = np.clip(s_boost, 0, 255).astype(np.uint8)

        hsv_boosted = cv2.merge([h, s_boost, v])
        result_bgr = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        return result_bgr


    def filter_background_by_saturation(self, image_bgr, s_thresh=40):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        # 掩码：保留高饱和度区域（过滤灰/黑/白背景）
        mask = cv2.inRange(s, s_thresh, 255)

        # 形态学处理：去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask_clean

    def enhance_contrast_opencv(self, image_bgr, alpha=1.0, beta=0):
        """
        alpha: 对比度，>1 增强，<1 减弱
        beta: 亮度调节，通常设为 0 不变
        """
        return cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)

    def detect_shapes(self, frame):
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.img_center = (self.width // 2, self.height // 2)
        img = self.enhance_contrast_opencv(frame)

        mask = self.filter_background_by_saturation(img)
        start_time = time.time()
        trapezoid_mask = detect_and_segment(frame, self.sam_predictor, self.yolo_model)
        end_time = time.time()
        print(f"[耗时] 检测和分割耗时: {end_time - start_time:.2f}秒")
        combined_mask = cv2.bitwise_or(mask, trapezoid_mask)

        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
        filtered_mask = cv2.medianBlur(closed, 7)
        edges = cv2.Canny(filtered_mask, 50, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        parallel_edge = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.width * self.height / 100: # 过滤小轮廓
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
                angle = (360 - self.compute_angle(shape, base_edge_tri[0], base_edge_tri[1])) % 120
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
                    angle = self.compute_angle(shape, diag_points[0], diag_points[1]) % 180
                    base_line = [tuple(diag_points[0]), tuple(diag_points[1])]
                else:
                    shape = "trapezoid"

                    # === 1. 找到最长边作为基准边 ===
                    max_len = 0
                    base_edge_tpd = None
                    for i in range(len(points)):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % len(points)]
                        length = np.linalg.norm(pt1 - pt2)
                        if length > max_len:
                            max_len = length
                            base_edge_tpd = [pt1, pt2]

                    # === 2. 使用基准边计算角度 ===
                    angle = self.compute_angle(shape, base_edge_tpd[0], base_edge_tpd[1])
                    base_line = [tuple(base_edge_tpd[0]), tuple(base_edge_tpd[1])]

                    for i in range(len(points)):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % len(points)]
                        is_same_edge = (
                            (np.array_equal(base_edge_tpd[0], pt1) and np.array_equal(base_edge_tpd[1], pt2)) or
                            (np.array_equal(base_edge_tpd[0], pt2) and np.array_equal(base_edge_tpd[1], pt1))
                        )
                        angle_edge = self.compute_angle(shape, pt1, pt2)
                        angle_diff = abs(angle - angle_edge)
                        if min(abs(angle_diff), abs(180 - angle_diff), abs(360 - angle_diff)) < 30 and not is_same_edge:
                            parallel_edge = [pt1, pt2]
                            break
                    
                    # === 4. 根据中点相对位置判断方向是否需要翻转 ===
                    if parallel_edge is not None:
                        print(f'origin angle:{angle}')
                        midpoint_base = (base_edge_tpd[0] + base_edge_tpd[1]) / 2
                        midpoint_parallel = (parallel_edge[0] + parallel_edge[1]) / 2
                        vec = midpoint_parallel - midpoint_base
                        if vec[0] > 0 and vec[1] > 0:
                            angle = (180 + angle) % 360
                            pass
                        elif vec[0] < 0 and vec[1] > 0:
                            angle = angle % 180
                            pass
                        else:
                            if vec[0] < 0 and abs(270 - angle) < 3:
                                angle = angle % 180
                                pass
                            elif vec[0] > 0 and abs(90 - angle) < 3:
                                angle = (180 + angle) % 360
                                pass
                            elif vec[1] > 0 and abs(angle) < 3:
                                angle = (180 + angle) % 360
                                pass
                            elif vec[1] < 0 and abs(180 - angle) < 3:
                                angle = (180 + angle) % 360
                                pass
                        print(f'processed angle:{angle}')


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

                angle = (360 - self.compute_angle(shape, base_edge_hex[0], base_edge_hex[1])) % 60
                base_line = [tuple(base_edge_hex[0]), tuple(base_edge_hex[1])]

            if shape != 'unknown':
                end_cx, end_cy = self.transform(cx, cy)
                distance = np.linalg.norm(np.array([cx, cy]) - np.array(self.img_center))
                results.append({
                    "shape": shape,
                    "center": (int(cx), int(cy)),
                    "distance": round(distance, 1),
                    "end_center":(round(end_cx, 4), round(end_cy, 4)),
                    "angle": round(angle, 1),
                    "contour": cnt,
                    "base_line": base_line,
                })

        return results
    
    def save_image(self, image, step_name, debug_dir):
        cv2.imwrite(f"{debug_dir}/{step_name}.png", image)

    def imshow(self, frame, result_strings, save= True, show= False, img_path = r'results'):
        output = frame.copy()
        for res in result_strings:
            cv2.drawContours(output, [res["contour"]], -1, (0,255,0), 2)
            cv2.circle(output, res["center"], 3, (0,0,255), -1)
            text = f"{res['shape']} {res['angle']}"
            cv2.putText(output, text, 
                    (res["center"][0]-40, res["center"][1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
            if "base_line" in res:
                pt1, pt2 = res["base_line"]
                cv2.line(output, pt1, pt2, (0, 0, 255), 2)
        if show:
            resized_output = cv2.resize(output, (int(self.width/2), int(self.height/2)))
            cv2.imshow("Result", resized_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            self.save_image(output, "final_result", img_path)
        return output

class ServerThread(QThread):
    timeout_signal = pyqtSignal()  # 用于通知主线程“超时了，可以重新点击按钮了”
    connected_signal = pyqtSignal() 

    def __init__(self, plc_server):
        super().__init__()
        self.plc_server = plc_server

    def run(self):
        self.plc_server.start_server()
        if self.plc_server.connected:
            self.connected_signal.emit()
        else:
            self.timeout_signal.emit()

class TcpImageServer:
    def __init__(self, host='192.168.1.100', port=2000, timeout=10):
        self.host = host
        self.port = port
        self.buffer_size = 2048
        self.timeout = timeout
        self.on_message_received = None  # 回调函数
        self.connected = False

    def handle_client(self):
        print(f"[连接] 来自 {self.addr}")
        while True:
            try:
                data = self.conn.recv(self.buffer_size)
                if not data:
                    print("[断开连接]")
                    break
                msg = data[2:].decode('utf-8', errors='ignore').strip()
                print(f"[接收] {msg}")
                if self.on_message_received:
                    self.on_message_received(msg)  # <- 调用回调

            except Exception as e:
                print(f"[异常] {e}")
                break
        self.conn.close()

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind((self.host, self.port))
        except socket.error as e:
            print(f"[错误] 未检测到此本地IP: {e}")
            return
        server.listen(1)
        server.settimeout(self.timeout) 
        print(f"[监听] 等待来自 PLC 的连接 {self.host}:{self.port}...等待 {self.timeout}s...")

        try:
            self.conn, self.addr = server.accept()
            self.connected = True
            thread = threading.Thread(target=self.handle_client)
            thread.start()
            # thread.join()
            # self.handle_client()
        except socket.timeout:
            self.connected = False
            print(f"[超时] 无连接")
        finally:
            server.close()

# ------------------- 图形绘制类 -----------------------

class CoordinateMapper:
    def __init__(self, origin_in_qt=QPointF(400, 200)):
        """
        :param origin_in_qt: 世界坐标系原点在 Qt 坐标系中的位置（QPointF）
        """
        self.origin = origin_in_qt  # Qt 坐标中的世界原点
        self.scale = 30 / 120   # 30 : 120px

    def qt_to_world(self, qt_point: QPointF) -> QPointF:
        """
        将 Qt 坐标点转换为世界坐标（x 左，y 下）
        """
        dx = (qt_point.x() - self.origin.x()) * self.scale
        dy = (qt_point.y() - self.origin.y()) * self.scale
        return QPointF(-dx, dy)  # x轴反转

    def world_to_qt(self, world_point: QPointF) -> QPointF:
        """
        将世界坐标转换为 Qt 坐标点
        """
        dx = -world_point.x() / self.scale
        dy = world_point.y() / self.scale
        return QPointF(self.origin.x() + dx, self.origin.y() + dy)

    def set_origin(self, new_origin: QPointF):
        """
        修改世界坐标原点在 Qt 中的位置
        """
        self.origin = new_origin

    def load_from_json(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            shapes = json.load(f)
        
        items = []

        for shape in shapes:
            world_x, world_y = shape["position"]
            p = QPointF(world_x, world_y)
            qt_point  = self.world_to_qt(p)

            item = DraggableShape(
                shape_type=shape["shape"],
                grid_size=self.grid_size if hasattr(self, 'grid_size') else 60  # 适配你的 grid_size
            )
            item.setPos(qt_point)
            item.set_angle(shape["angle"])
            items.append(item)
        return items

class GridScene(QGraphicsScene):
    def __init__(self, x, y, w, h, grid_size=20):
        super().__init__(x, y, w, h)
        self.mapper = CoordinateMapper()
        self.grid_size = grid_size
        self.axis_pen_x = QPen(Qt.red, 2)
        self.arrow_brush_x = QBrush(Qt.red)
        self.axis_pen_y = QPen(Qt.green, 2)
        self.arrow_brush_y = QBrush(Qt.green)
        self.draw_grid()
        self.draw_coordinate_axes()
        self.draw_world()

    def draw_world(self):
        origin = self.mapper.origin
        arrow_length = 20
        x_end = origin + QPointF(-arrow_length, 0)
        y_end = origin + QPointF(0, arrow_length)

        # 绘制 X 轴线并置顶
        x_line = self.addLine(origin.x(), origin.y(), x_end.x(), x_end.y(), QPen(Qt.darkGreen, 2))
        x_line.setZValue(1000)

        # 绘制 X 轴文字并置顶
        x_text = self.addText("X")
        x_text.setPos(x_end.x() - 10, x_end.y())
        x_text.setZValue(1000)

        # 绘制 Y 轴线并置顶
        y_line = self.addLine(origin.x(), origin.y(), y_end.x(), y_end.y(), QPen(Qt.darkBlue, 2))
        y_line.setZValue(1000)

        # 绘制 Y 轴文字并置顶
        y_text = self.addText("Y")
        y_text.setPos(y_end.x() + 2, y_end.y())
        y_text.setZValue(1000)



    def draw_grid(self):
        for x in range(0, int(self.width()), self.grid_size):
            self.addLine(x, 0, x, self.height(), QPen(Qt.lightGray, 0.5))
        for y in range(0, int(self.height()), self.grid_size):
            self.addLine(0, y, self.width(), y, QPen(Qt.lightGray, 0.5))

    def draw_coordinate_axes(self):
        # 坐标轴起点
        origin = QPointF(0, 0)
        x_end = QPointF(20, 0)
        y_end = QPointF(0, 20)

        # 绘制 X 轴和 Y 轴线条
        self.addLine(origin.x(), origin.y(), x_end.x(), x_end.y(), self.axis_pen_x)
        self.addLine(origin.x(), origin.y(), y_end.x(), y_end.y(), self.axis_pen_y)

        # 绘制箭头（X 轴）
        x_arrow = QPolygonF([
            QPointF(x_end.x(), x_end.y()),
            QPointF(x_end.x() - 6, x_end.y() - 4),
            QPointF(x_end.x() - 6, x_end.y() + 4)
        ])
        self.addPolygon(x_arrow, self.axis_pen_x, self.arrow_brush_x)

        # 绘制箭头（Y 轴）
        y_arrow = QPolygonF([
            QPointF(y_end.x(), y_end.y()),
            QPointF(y_end.x() - 4, y_end.y() - 6),
            QPointF(y_end.x() + 4, y_end.y() - 6)
        ])
        self.addPolygon(y_arrow, self.axis_pen_y, self.arrow_brush_y)

class DraggableShape(QGraphicsItem):
    def __init__(self, shape_type, grid_size= 20):
        super().__init__()
        self.shape_type = shape_type
        try:
            self.color = {
                "triangle": QColor("#e07b39"),
                "diamond": QColor("#c2356b"),
                "trapezoid": QColor("#1c5d4a"),
                "hexagon": QColor("#f08c1d")
            }.get(shape_type, Qt.black)
        except:
            self.color = Qt.black
        self.angle = 0
        self._right_mouse_press_x = None
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        if shape_type == "triangle":
            h = grid_size * 2 * (3 ** 0.5) / 2
            points = [QPointF(0, 0), QPointF(grid_size * 2, 0), QPointF(grid_size, h)]
        elif shape_type == "diamond":
            h = grid_size * 2 * (3 ** 0.5) / 2
            points = [QPointF(h, 0), QPointF(2*h, grid_size), QPointF(h, grid_size * 2), QPointF(0, grid_size)]
        elif shape_type == "trapezoid":
            h = grid_size * 2 * (3 ** 0.5) / 2
            points = [QPointF(0, 0), QPointF(grid_size * 2, 0), QPointF(grid_size*3, h), QPointF(-grid_size, h)]
        elif shape_type == "hexagon":
            points = [QPointF(grid_size, 0), QPointF(3*grid_size, 0), QPointF(4*grid_size, grid_size * 3 ** 0.5),
                      QPointF(3*grid_size, 2*grid_size * 3 ** 0.5), QPointF(grid_size, 2*grid_size * 3 ** 0.5), QPointF(0, grid_size * 3 ** 0.5)]
        else:
            raise ValueError("Unknown shape type")

        cx, cy = self.calc_center(points)
        polygon = QPolygonF([QPointF(p.x() - cx, p.y() - cy) for p in points])
        self.polygon = polygon
        self.shape = QGraphicsPolygonItem(polygon, parent=self)
        self.shape.setBrush(QBrush(self.color))
        self.setTransformOriginPoint(0, 0)

    def calc_center(self, points):
        x = sum(p.x() for p in points) / len(points)
        y = sum(p.y() for p in points) / len(points)
        return x, y

    def boundingRect(self):
        return self.shape.boundingRect()

    def paint(self, painter, option, widget):
        pass

    def set_angle(self, angle):
        self.angle = angle % 360
        self.setRotation(-self.angle)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._right_mouse_press_x = event.scenePos().x()
            self.setSelected(True)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            old_pos = self.pos()  # 记录原位置
            super().mouseMoveEvent(event)
            if not self.within_bounds():
                self.setPos(old_pos)  # 撤销越界移动
        elif event.buttons() == Qt.RightButton:
            current_x = event.scenePos().x()
            delta = current_x - self._right_mouse_press_x
            step = 5
            if abs(delta) >= 2:
                new_angle = self.angle + (-step if delta > 0 else step)
                if self.can_rotate_to(new_angle):
                    self.set_angle(new_angle)
                # 否则不旋转
                self._right_mouse_press_x = current_x
            event.accept()

    def within_bounds(self):
        # 当前场景变换后图形边界是否还在 0,0~600,600 范围内
        scene_rect = QRectF(0, 0, 600, 600)
        return scene_rect.contains(self.mapToScene(self.boundingRect()).boundingRect())

    def can_rotate_to(self, angle):
        transform = QTransform()
        transform.rotate(-angle % 360)
        rotated_polygon = transform.map(self.polygon)
        rotated_rect = rotated_polygon.boundingRect()
        scene_rect = QRectF(0, 0, 600, 600)
        pos = self.pos()
        translated_rect = rotated_rect.translated(pos)
        return scene_rect.contains(translated_rect)



    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.scene().removeItem(self)

# ------------------- 主界面类 -----------------------
class ShapeDesignerMainWindow(QMainWindow):
    message_received = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.message_received.connect(self.update_plc_display)
        self.setWindowTitle("图形设计配置界面")
        self.resize_to_screen()

        # 图像检测器
        self.detector = ImgDetector()
        # 相机
        self.camera = None
        #配置结果
        self.shapes_data = []

        #当前的图形索引
        self.shape_idx = 0
        #已匹配的索引集合
        self.matched_indices = set()

        # 左侧图形列表区域
        self.list_widget = QListWidget()
        shape_names = [("正三角形", "triangle"), ("菱形", "diamond"),
                       ("梯形", "trapezoid"), ("正六边形", "hexagon")]
        self.shape_map = {}
        for name, code in shape_names:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(80, 40))
            self.list_widget.addItem(item)
            self.shape_map[name] = code
        self.list_widget.itemDoubleClicked.connect(self.add_shape_from_list)
        self.save_config_button = QPushButton("保存配置")
        self.save_config_button.clicked.connect(self.save_config)
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.setStyleSheet("font-size: 14px; padding: 6px;")
        self.load_config_btn.clicked.connect(self.load_config)

        self.clear_btn = QPushButton("重置")
        self.clear_btn.setStyleSheet("font-size: 14px; padding: 6px;")
        self.clear_btn.clicked.connect(self.clear_all)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.save_config_button)
        left_layout.addWidget(self.load_config_btn)
        left_layout.addWidget(self.clear_btn)

        # 中间绘图区域
        self.scene = GridScene(0, 0, 600, 600, grid_size=60)
        self.view = QGraphicsView(self.scene)
        self.scene.addRect(QRectF(0, 0, 600, 600), QPen(QColor("gray"), 1, Qt.DashLine))

        self.x0_input = QSpinBox()
        self.x0_input.setRange(0, 500)
        self.x0_input.setValue(400)
        self.x0_input.setPrefix("x0: ")

        self.y0_input = QSpinBox()
        self.y0_input.setRange(0, 500)
        self.y0_input.setValue(200)
        self.y0_input.setPrefix("y0: ")

        # 原点设置按钮
        self.set_origin_btn = QPushButton("设置世界原点")
        self.set_origin_btn.setStyleSheet("font-size: 14px; padding: 6px;")
        self.set_origin_btn.clicked.connect(self.set_origin)

        self.tip_label = QLabel("提示：双击左侧图形以添加")
        self.tip_label.setStyleSheet("color: gray; font-size: 14px;")
        self.btn_send = QPushButton("提交配置")
        self.btn_send.setStyleSheet("font-size: 14px; padding: 6px;")
        self.btn_send.clicked.connect(self.send_layout)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.addWidget(self.view)
        center_layout.addWidget(self.tip_label)
        center_layout.addWidget(self.btn_send)
        center_layout.addWidget(self.x0_input)
        center_layout.addWidget(self.y0_input)
        center_layout.addWidget(self.set_origin_btn)

        # 右侧功能区域
        self.btn_detect = QPushButton("运行图像检测")
        self.btn_detect.setStyleSheet("font-size: 14px; padding: 6px;")
        self.btn_detect.clicked.connect(self.update_detection_result)

        self.btn_camera = QPushButton("连接相机")
        self.btn_camera.setStyleSheet("font-size: 14px; padding: 6px;")
        self.btn_camera.clicked.connect(self.connect_camera)

        config_group = QGroupBox("TCP配置")
        config_layout = QGridLayout()
        config_layout.addWidget(QLabel("主机IP:"), 0, 0)
        self.ip_input = QComboBox()
        self.ip_input.addItems(get_local_ips())
        config_layout.addWidget(self.ip_input, 0, 1)
        config_layout.addWidget(QLabel("端口:"), 1, 0)
        self.port_input = QLineEdit("2000")
        config_layout.addWidget(self.port_input, 1, 1)

        self.btn_connect = QPushButton("连接PLC")
        self.btn_connect.clicked.connect(self.connect_plc)
        config_layout.addWidget(self.btn_connect, 2, 0, 1, 2)
        config_group.setLayout(config_layout)

        self.image_label = QLabel("图像检测结果显示区域")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background: #f0f0f0;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plc_data_display = QTextEdit()
        self.plc_data_display.setReadOnly(True)
        self.plc_data_display.setPlaceholderText("PLC通信数据将在此显示...")
        self.plc_data_display.setFixedHeight(120)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.btn_detect)
        right_layout.addWidget(self.btn_camera)
        right_layout.addWidget(config_group)
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.plc_data_display)

        # Splitter 三栏自由拖拽
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        # 设置主窗口中心部件
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

        self.shapes = []
        self.current_qpixmap = None

    def resize_to_screen(self):
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.resize(width, height)

    def add_shape_from_list(self, item):
        shape_type = self.shape_map.get(item.text(), "triangle")
        shape = DraggableShape(shape_type, self.scene.grid_size)
        shape.setPos(100, 100)
        self.scene.addItem(shape)
        self.shapes.append(shape)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene.selectedItems():
                self.scene.removeItem(item)
                if item in self.shapes:
                    self.shapes.remove(item)
                # 删除具有相同 item_id 的 shape_data
                self.shapes_data = [
                    data for data in self.shapes_data if data.get("item_id") != id(item)
                ]
            event.accept()
        else:
            super().keyPressEvent(event)

    def send_layout(self):
        for item in self.shapes:
            angle = item.angle
            w_point = self.scene.mapper.qt_to_world(item)
            if item.shape_type == "triangle":
                angle = angle % 120
            elif item.shape_type == "diamond":
                angle = angle % 180
            elif item.shape_type == "hexagon":
                angle = angle % 60
            shape_data = {
                "shape": item.shape_type,
                "center": (round(w_point.x(), 4), round(w_point.y(), 4)),
                "angle": round(angle, 1),
                "item_id": id(item)  # 添加唯一标识符           
            }
            if not any(data.get("item_id") == shape_data["item_id"] for data in self.shapes_data):
                self.shapes_data.append(shape_data)

        json_data = json.dumps(self.shapes_data)
        info = f"[准备发送至PLC {self.ip_input.currentText()}:{self.port_input.text()}]\n{json_data}\n"
        self.plc_data_display.append(info)

    def set_origin(self):
        x0 = self.x0_input.value()
        y0 = self.y0_input.value()
        self.scene.mapper.origin = QPointF(x0, y0)
        self.scene.clear()
        self.scene.draw_grid()
        self.scene.draw_coordinate_axes()
        self.scene.draw_world()

    def load_config(self):
        # 弹出文件选择框，选择 JSON 配置文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "config", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return  # 用户取消选择
        try:
            items = self.scene.mapper.load_from_json(file_path)
            for item in items:
                self.scene.addItem(item)
                self.shapes.append(item)
        except Exception as e:
            self.plc_data_display.append(f"无法加载配置文件：\n{str(e)}")

    def save_config(self):
        config_list = []
        for shape in self.shapes_data:
            config_list.append({
                "position": shape["center"],
                "shape": shape["shape"],
                "angle": shape["angle"],
            })

        file_path, _ = QFileDialog.getSaveFileName(self, "保存配置文件", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_list, f, ensure_ascii=False, indent=2)
        self.plc_data_display.append(f"[保存配置] 已保存到 {file_path}")

    def clear_all(self):
        for item in self.shapes:
            self.scene.removeItem(item)
        self.shapes.clear()
        self.shapes_data.clear()
        self.matched_indices.clear()

    def connect_plc(self):
        ip = self.ip_input.currentText()
        port = self.port_input.text()
        msg = f"尝试连接到PLC: {ip}:{port}"
        self.plc_data_display.append(msg)

        # 禁用按钮防止重复点击
        self.btn_connect.setEnabled(False)

        # 启动 TCP Server
        self.plc = TcpImageServer(host=ip, port=int(port.strip()), timeout=10)
        self.plc.on_message_received = self.msg_handler

        # 创建后台线程运行 start_server
        self.server_thread = ServerThread(self.plc)
        self.server_thread.timeout_signal.connect(self.on_server_timeout)
        self.server_thread.connected_signal.connect(self.on_server_connected)
        self.server_thread.start()

    def on_server_timeout(self):
        self.plc_data_display.append("[提示] 连接超时，未检测到 PLC。")
        self.btn_connect.setEnabled(True)

    def on_server_connected(self):
        self.plc_data_display.append("[提示] 连接成功！")
        self.btn_connect.setEnabled(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_image_label_size()

    def adjust_image_label_size(self):
        target_ratio = 2592 / 1944
        max_width = self.width() * 0.45
        max_height = self.height() * 0.75

        if max_width / max_height > target_ratio:
            new_height = max_height
            new_width = new_height * target_ratio
        else:
            new_width = max_width
            new_height = new_width / target_ratio

        self.image_label.setFixedSize(int(new_width), int(new_height))

    def update_detection_result(self):
        if self.camera is not None:
            frame = self.camera.get_img()
            while frame is None:
                frame = self.camera.get_img()
        else:
            frame = cv2.imread(self.detector.img_path)
            self.plc_data_display.append("[提示] 未检测到相机，执行示例图片检测！")
            
        try:
            results = self.detector.detect_shapes(frame)
        except TypeError as e:
            self.plc_data_display.append("[警告] 模型加载中······")
            return []

        results.sort(key=lambda x: x["distance"])
        format_results = []
        image = self.detector.imshow(frame, results, save=False, show=False)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignCenter)

        for res in results:
            format_results.append({
                "shape": res['shape'],
                "center": res['end_center'],
                "angle": res['angle'],
            })
        return format_results
    
    def connect_camera(self):
        try:
            self.camera = Camera(0)
            self.btn_camera.setEnabled(False)
        except:
            self.plc_data_display.append("[提示] 未检测到相机。")
            self.btn_camera.setEnabled(True)
    
    def update_plc_display(self, msg):
        self.plc_data_display.append(msg)


    def msg_handler(self, msg):
        shape_map = {
                "triangle": "S",
                "diamond": "P",
                "hexagon": "L",
                "trapezoid": "T"
            }
        self.message_received.emit(f"[来自plc]: {msg}")
        if msg[:6] == "GETNUM":
            sleep(0.5)  # 等待相机稳定
            first_scan = self.update_detection_result()
            self.shape_pair_list = match(first_scan, self.shapes_data)
            self.num_dic = count_shapes(first_scan) 
            message = generate_vision_string(self.num_dic)
            self.message_received.emit("[发送plc]:" + message)
            self.plc.conn.sendall(message.encode('utf-8'))
        elif msg[:3] in [f"+{i}" for i in range(11, 22)]:
            # 是 +1 到 +11 之间的字符串
            shape_pair = self.shape_pair_list[self.shape_idx]
            if shape_pair["canvas"] is None:
                self.message_received.emit(f"[警告] 未能匹配 shape: {shape_pair['result']['shape']}，坐标: {shape_pair['result']['center']}")
                return
            shape_code = shape_map.get(shape_pair['result']['shape'], "U")  # U = 未知
            x, y = shape_pair['result']['center']
            r = 180 - (shape_pair['canvas']['angle'] - shape_pair['result']['angle'])
            message = f"00OK,{shape_code},x{round(x,3)},y{round(y,3)},r{round(r,3)}*"
            self.message_received.emit("[发送plc]:" + message)
            self.plc.conn.sendall(message.encode('utf-8'))
            self.shape_idx += 1
            if self.shape_idx >= sum(self.num_dic.values()):
                self.shape_idx = 0
                return

        elif msg[:3] in [f"+{i}" for i in range(51, 55)]:
            if not hasattr(self, '_last_msg_time'):
                self._last_msg_time = {}
            current_time = time.time()
            if msg[:3] in self._last_msg_time and current_time - self._last_msg_time[msg[:3]] < 1:
                return  # Ignore if the message is received within 1 seconds of the last one

            self._last_msg_time[msg[:3]] = current_time
            sleep(0.1)  # 等待相机稳定
            second_scan = self.update_detection_result()
            found_result = None
            for item in second_scan:
                if item['shape'] == 'triangle' and msg[:3] == '+51':
                    found_result = item
                    break
                elif item['shape'] == 'diamond' and msg[:3] == '+52':
                    found_result = item
                    break
                elif item['shape'] == 'hexagon' and msg[:3] == '+53':
                    found_result = item
                    break
                elif item['shape'] == 'trapezoid' and msg[:3] == '+54':
                    found_result = item
                    break

            if found_result:
                shape_pair = match_one(found_result, self.shapes_data, self.matched_indices)
                if shape_pair["canvas"] is not None:
                    shape_code = shape_map.get(shape_pair['result']['shape'], "U")
                    x, y = shape_pair['result']['center']
                    X, Y = shape_pair['canvas']['center']
                    r = (shape_pair['canvas']['angle'] - shape_pair['result']['angle'] + 180) *5 # plc角度需要*5
                    message = f"00OK,{shape_code},x{round(x,3)},y{round(y,3)},r{round(r,3)},X{round(X,3)},Y{round(Y,3)}*"
                    self.message_received.emit("[发送plc]:" + message)
                    self.plc.conn.sendall(message.encode('utf-8'))
                else:
                    print("匹配失败")
            else:
                print("未找到对应图形")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShapeDesignerMainWindow()
    window.show()
    sys.exit(app.exec_())


