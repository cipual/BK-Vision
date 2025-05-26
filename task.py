import socket
import threading
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPolygonItem, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
    QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QPolygonF, QBrush, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF, QSize
import json

class ImgDetector:
    def __init__(self, img_path='imgs/paizhao.png', txt_path='config\waican.txt'):
        self.img_path = img_path
        # self.Trans = []
        self.Trans = np.array([[-0.072726, -0.000536, 86.089027],
                            [-0.000666, 0.072689, -79.587906],
                            [0.000000, 0.000000, 1.000000]])
        self.cameraMatrix = np.array([
                        [3098.393199379014, 0, 1216.273938824572],
                        [0, 3099.618345026281, 1074.290301250453],
                        [0, 0, 1]]
                        )
        # with open(txt_path, 'r') as f:
        #     for line in f:
        #         parts = [float(x.strip()) for x in line.strip().split(',') if x.strip()]
        #         self.Trans.append(parts)
        # self.Trans = np.array(self.Trans)  # 3x3，列主序
        # self.Trans = self.Trans.T  

    def compute_angle(self, pt1, pt2):
        if pt1[0] > pt2[0]:
            pt1, pt2 = pt2, pt1
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        return np.degrees(np.arctan2(dy, dx)) % 180
    
    def transform(self, x, y):
        point_h = np.array([x, y, 1])  # 齐次坐标
        transformed = self.Trans @ point_h
        return transformed[0], transformed[1]

    def detect_shapes(self, frame):
        # self.image = cv2.imread(self.img_path)
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.img_center = (self.width // 2, self.height // 2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

        color_ranges = {
            "red": ([150, 50, 40], [180, 200, 255]),
            "yellow": [(5, 80, 60), (30, 255, 255)],
            "orange": [(5, 150, 60), (25, 255, 180)],
            "green": [(35, 53, 28), (88, 94, 75)],
        }

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges.values():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
        filtered_mask = cv2.medianBlur(closed, 7)
        edges = cv2.Canny(filtered_mask, 50, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
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
                angle = (360 - self.compute_angle(base_edge_tri[0], base_edge_tri[1])) % 120
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
                    angle = 180 - self.compute_angle(diag_points[0], diag_points[1])
                    base_line = [tuple(diag_points[0]), tuple(diag_points[1])]
                else:
                    shape = "trapezoid"
                    # 找到最长边作为基准边
                    max_len = 0
                    base_edge_tpd = None
                    for i in range(len(points)):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % len(points)]  # 邻接点构成边
                        length = np.linalg.norm(pt1 - pt2)
                        if length > max_len:
                            max_len = length
                            base_edge_tpd = [pt1, pt2]

                    angle = 360 - self.compute_angle(base_edge_tpd[0], base_edge_tpd[1])
                    base_line = [tuple(base_edge_tpd[0]), tuple(base_edge_tpd[1])]

                    # === 更精确地寻找“平行短边” ===
                    # 计算每条边的长度
                    edges = []
                    for i in range(len(points)):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % len(points)]
                        length = np.linalg.norm(pt1 - pt2)
                        edges.append((length, [pt1, pt2]))

                    # 按长度排序，最长的为基准边，次长的为平行短边
                    edges.sort(key=lambda x: x[0], reverse=True)
                    base_edge_tpd = edges[0][1]
                    parallel_edge = edges[1][1]

                    # 重新计算基准边和平行边的中点
                    midpoint_base = (base_edge_tpd[0] + base_edge_tpd[1]) / 2
                    midpoint_parallel = (parallel_edge[0] + parallel_edge[1]) / 2

                    # 判断相对位置，确保角度方向一致
                    vec = midpoint_parallel - midpoint_base
                    if vec[0] < 0:  # 如果平行边在基准边左侧，调整角度
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

                angle = (360 - self.compute_angle(base_edge_hex[0], base_edge_hex[1])) % 60
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
    
    def encoder():
        pass

class TcpImageServer:
    def __init__(self, host='192.168.1.5', port=2000, detector=None):
        self.host = host
        self.port = port
        self.buffer_size = 2048
        self.detector = detector
        self.detection_results = []
        self.result_index = 0

    def handle_client(self, conn, addr):
        print(f"[连接] 来自 {addr}")
        while True:
            try:
                data = conn.recv(self.buffer_size)
                if not data:
                    print("[断开连接]")
                    break
                msg = data[2:].decode('utf-8', errors='ignore').strip()
                print(f"[接收] {msg}")

                if msg == "Start":
                    self.detection_results = self.detector.detect_shapes()
                    self.result_index = 0
                    print(f"[检测完成] 共检测到 {len(self.detection_results)} 个目标。")

                elif msg[:2] == "OK":
                    if self.result_index < len(self.detection_results):
                        response = self.detection_results[self.result_index]
                        print(f"[发送] {response}")
                        conn.sendall(response.encode('utf-8'))
                        self.result_index += 1
                    else:
                        print("[发送] 所有数据已发送完毕")

                elif msg[:4] == "Stop":
                    print("[停止] 清除当前识别结果")
                    self.detection_results = []
                    self.result_index = 0

            except Exception as e:
                print(f"[异常] {e}")
                break
        conn.close()

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"[监听] 等待来自 PLC 的连接 {self.host}:{self.port}...")

        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()


# class GridScene(QGraphicsScene):
#     def __init__(self, *args, grid_size=20, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.grid_size = grid_size

#     def drawBackground(self, painter, rect):
#         left = int(rect.left()) - (int(rect.left()) % self.grid_size)
#         top = int(rect.top()) - (int(rect.top()) % self.grid_size)

#         lines = []
#         # 竖线
#         x = left
#         while x < rect.right():
#             lines.append(QLineF(x, rect.top(), x, rect.bottom()))
#             x += self.grid_size
#         # 横线
#         y = top
#         while y < rect.bottom():
#             lines.append(QLineF(rect.left(), y, rect.right(), y))
#             y += self.grid_size

#         painter.setPen(QPen(QColor("#ddd"), 0))
#         painter.drawLines(lines)


# class DraggableShape(QGraphicsItem):
#     def __init__(self, shape_type, color=Qt.red):
#         super().__init__()
#         self.shape_type = shape_type
#         self.color = color
#         self.angle = 0
#         self._right_mouse_press_x = None
#         self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
#         self.setAcceptHoverEvents(True)

#         if shape_type == "triangle":
#             h = 40 * (3 ** 0.5) / 2
#             points = [QPointF(0, 0), QPointF(40, 0), QPointF(20, h)]
#         elif shape_type == "diamond":
#             h = 40 * (3 ** 0.5) / 2
#             points = [QPointF(h, 0), QPointF(2*h, 20), QPointF(h, 40), QPointF(0, 20)]
#         elif shape_type == "trapezoid":
#             h = 40 * (3 ** 0.5) / 2
#             points = [QPointF(0, 0), QPointF(40, 0), QPointF(60, h), QPointF(-20, h)]
#         elif shape_type == "hexagon":
#             a = 20
#             points = [QPointF(a, 0), QPointF(3*a, 0), QPointF(4*a, a * 3 ** 0.5),
#                     QPointF(3*a, 2*a * 3 ** 0.5), QPointF(a, 2*a * 3 ** 0.5), QPointF(0, a * 3 ** 0.5)]
#         else:
#             raise ValueError("Unknown shape type")

#         # 顶点坐标平移，使几何中心为原点
#         cx, cy = self.calc_center(points)
#         polygon = QPolygonF([QPointF(p.x() - cx, p.y() - cy) for p in points])

#         self.polygon = polygon
#         self.shape = QGraphicsPolygonItem(polygon, parent=self)
#         self.shape.setBrush(QBrush(self.color))
#         self.setTransformOriginPoint(0, 0)


#     def boundingRect(self):
#         return self.shape.boundingRect()
    
#     def calc_center(self, points):
#         x = sum(p.x() for p in points) / len(points)
#         y = sum(p.y() for p in points) / len(points)
#         return x, y

#     def paint(self, painter, option, widget):
#         pass

#     def set_angle(self, angle):
#         self.angle = angle % 360
#         self.setRotation(-self.angle)

#     def mousePressEvent(self, event):
#         if event.button() == Qt.RightButton:
#             self._right_mouse_press_x = event.scenePos().x()
#             self.setSelected(True)
#             event.accept()
#         else:
#             super().mousePressEvent(event)


#     def mouseReleaseEvent(self, event):
#         super().mouseReleaseEvent(event)
#         # grid_size = 20  # 同上
#         # pos = self.pos()
#         # x = round(pos.x() / grid_size) * grid_size
#         # y = round(pos.y() / grid_size) * grid_size
#         # self.setPos(x, y)


#     def mouseMoveEvent(self, event):
#         if event.buttons() == Qt.LeftButton:
#             super().mouseMoveEvent(event)
#         elif event.buttons() == Qt.RightButton:
#             current_x = event.scenePos().x()
#             delta = current_x - self._right_mouse_press_x
#             step = 5  # 旋转步长5度
#             if abs(delta) >= 2:  # 限制每次旋转触发的最小移动距离，防止太灵敏
#                 if delta > 0:
#                     self.angle -= step
#                 else:
#                     self.angle += step
#                 self.angle %= 360
#                 self.set_angle(self.angle)
#                 self._right_mouse_press_x = current_x
#             event.accept()


#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Delete:
#             self.scene().removeItem(self)

# class ShapeDesigner(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("图形摆放设计器")

#         layout = QHBoxLayout(self)

#         self.list_widget = QListWidget()
#         shape_names = [
#             ("正三角形", "triangle"),
#             ("菱形", "diamond"),
#             ("梯形", "trapezoid"),
#             ("正六边形", "hexagon")
#         ]
#         self.shape_map = {}
#         for display_name, code_name in shape_names:
#             item = QListWidgetItem(display_name)
#             item.setSizeHint(QSize(80, 40))
#             self.list_widget.addItem(item)
#             self.shape_map[display_name] = code_name

#         self.list_widget.itemDoubleClicked.connect(self.add_shape_from_list)

#         # 添加提示标签
#         self.tip_label = QLabel("提示：双击左侧图形以添加")
#         self.tip_label.setStyleSheet("color: gray; font-size: 12px;")

#         right_layout = QVBoxLayout()
#         self.scene = GridScene(0, 0, 500, 500, grid_size=20)
#         self.view = QGraphicsView(self.scene)
#         self.view.setRenderHint(QPainter.Antialiasing)

#         # 添加虚线边框
#         pen = QPen(QColor("gray"), 1, Qt.DashLine)
#         self.scene.addRect(QRectF(0, 0, 500, 500), pen)

#         self.btn_send = QPushButton("发送配置")
#         self.btn_send.clicked.connect(self.send_layout)

#         right_layout.addWidget(self.view)
#         right_layout.addWidget(self.tip_label)
#         right_layout.addWidget(self.btn_send)

#         layout.addWidget(self.list_widget)
#         layout.addLayout(right_layout)

#         self.shapes = []

#     def add_shape_from_list(self, item):
#         shape_type = self.shape_map.get(item.text(), "triangle")
#         color = {
#             "triangle": QColor("#e07b39"),    # 橙色三角形
#             "diamond": QColor("#c2356b"),     # 紫红菱形
#             "trapezoid": QColor("#1c5d4a"),   # 墨绿色梯形
#             "hexagon": QColor("#f08c1d")      # 橙黄色六边形
#         }.get(shape_type, Qt.black)
#         shape = DraggableShape(shape_type, color)
#         shape.setPos(100, 100)
#         self.scene.addItem(shape)
#         self.shapes.append(shape)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Delete:
#             for item in self.scene.selectedItems():
#                 self.scene.removeItem(item)
#                 if item in self.shapes:
#                     self.shapes.remove(item)
#             event.accept()
#         else:
#             super().keyPressEvent(event)

#     def send_layout(self):
#         shapes_data = []
#         for item in self.shapes:
#             if item.shape_type == "triangle":
#                 angle = item.angle % 120
#             elif item.shape_type == "diamond":
#                 angle = item.angle % 180
#             elif item.shape_type == "hexagon":
#                 angle = item.angle % 60
#             elif item.shape_type == "trapezoid":
#                 angle = item.angle
#             shape_data = {
#                 "shape": item.shape_type,
#                 "x": int(item.x()),
#                 "y": int(item.y()),
#                 "angle": round(angle, 1),
#             }
#             shapes_data.append(shape_data)

#         json_data = json.dumps(shapes_data)
#         print("发送数据:", json_data)



if __name__ == '__main__':
    img_path = r'imgs\20250526_221623.jpg'
    detector = ImgDetector(img_path=img_path, txt_path='config\waican.txt')
    image = cv2.imread(detector.img_path)
    results = detector.detect_shapes(image)
    detector.imshow(image, results, False, True)
    # server = TcpImageServer(detector=detector)
    # server.start_server()
