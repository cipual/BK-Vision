import sys
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPolygonItem, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
    QListWidget, QListWidgetItem, QTextEdit
)
from PyQt5.QtGui import QPolygonF, QBrush, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF, QSize
import json

class GridScene(QGraphicsScene):
    def __init__(self, *args, grid_size=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size

    def drawBackground(self, painter, rect):
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)

        lines = []
        # 竖线
        x = left
        while x < rect.right():
            lines.append(QLineF(x, rect.top(), x, rect.bottom()))
            x += self.grid_size
        # 横线
        y = top
        while y < rect.bottom():
            lines.append(QLineF(rect.left(), y, rect.right(), y))
            y += self.grid_size

        painter.setPen(QPen(QColor("#ddd"), 0))
        painter.drawLines(lines)


class DraggableShape(QGraphicsItem):
    def __init__(self, shape_type, color=Qt.red):
        super().__init__()
        self.shape_type = shape_type
        self.color = color
        self.angle = 0
        self._right_mouse_press_x = None
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        if shape_type == "triangle":
            h = 40 * (3 ** 0.5) / 2
            points = [QPointF(0, 0), QPointF(40, 0), QPointF(20, h)]
        elif shape_type == "diamond":
            h = 40 * (3 ** 0.5) / 2
            points = [QPointF(h, 0), QPointF(2*h, 20), QPointF(h, 40), QPointF(0, 20)]
        elif shape_type == "trapezoid":
            h = 40 * (3 ** 0.5) / 2
            points = [QPointF(0, 0), QPointF(40, 0), QPointF(60, h), QPointF(-20, h)]
        elif shape_type == "hexagon":
            a = 20
            points = [QPointF(a, 0), QPointF(3*a, 0), QPointF(4*a, a * 3 ** 0.5),
                    QPointF(3*a, 2*a * 3 ** 0.5), QPointF(a, 2*a * 3 ** 0.5), QPointF(0, a * 3 ** 0.5)]
        else:
            raise ValueError("Unknown shape type")

        # 顶点坐标平移，使几何中心为原点
        cx, cy = self.calc_center(points)
        polygon = QPolygonF([QPointF(p.x() - cx, p.y() - cy) for p in points])

        self.polygon = polygon
        self.shape = QGraphicsPolygonItem(polygon, parent=self)
        self.shape.setBrush(QBrush(self.color))
        self.setTransformOriginPoint(0, 0)


    def boundingRect(self):
        return self.shape.boundingRect()
    
    def calc_center(self, points):
        x = sum(p.x() for p in points) / len(points)
        y = sum(p.y() for p in points) / len(points)
        return x, y

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


    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # grid_size = 20  # 同上
        # pos = self.pos()
        # x = round(pos.x() / grid_size) * grid_size
        # y = round(pos.y() / grid_size) * grid_size
        # self.setPos(x, y)


    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            super().mouseMoveEvent(event)
        elif event.buttons() == Qt.RightButton:
            current_x = event.scenePos().x()
            delta = current_x - self._right_mouse_press_x
            step = 5  # 旋转步长5度
            if abs(delta) >= 2:  # 限制每次旋转触发的最小移动距离，防止太灵敏
                if delta > 0:
                    self.angle -= step
                else:
                    self.angle += step
                self.angle %= 360
                self.set_angle(self.angle)
                self._right_mouse_press_x = current_x
            event.accept()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.scene().removeItem(self)

class ShapeDesigner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图形摆放设计器")

        main_layout = QHBoxLayout(self)

        # 左侧：图像窗口 + 数据窗口 + 连接按钮
        left_layout = QVBoxLayout()

        self.image_label = QLabel("图像结果显示区域")
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray;")

        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        self.data_text.setFixedSize(200, 200)
        self.data_text.setStyleSheet("border: 1px solid gray;")
        self.data_text.setPlaceholderText("PLC通信数据将在此显示...")

        self.btn_connect = QPushButton("连接PLC")
        self.btn_connect.clicked.connect(self.connect_to_plc)

        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.data_text)
        left_layout.addWidget(self.btn_connect)

        # 中间：图形选择列表
        self.list_widget = QListWidget()
        shape_names = [
            ("正三角形", "triangle"),
            ("菱形", "diamond"),
            ("梯形", "trapezoid"),
            ("正六边形", "hexagon")
        ]
        self.shape_map = {}
        for display_name, code_name in shape_names:
            item = QListWidgetItem(display_name)
            item.setSizeHint(QSize(80, 40))
            self.list_widget.addItem(item)
            self.shape_map[display_name] = code_name
        self.list_widget.itemDoubleClicked.connect(self.add_shape_from_list)

        # 右侧：场景和控制按钮
        right_layout = QVBoxLayout()
        self.scene = GridScene(0, 0, 500, 500, grid_size=20)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor("gray"), 1, Qt.DashLine)
        self.scene.addRect(QRectF(0, 0, 500, 500), pen)

        self.tip_label = QLabel("提示：双击左侧图形以添加")
        self.tip_label.setStyleSheet("color: gray; font-size: 12px;")
        self.btn_send = QPushButton("发送配置")
        self.btn_send.clicked.connect(self.send_layout)

        right_layout.addWidget(self.view)
        right_layout.addWidget(self.tip_label)
        right_layout.addWidget(self.btn_send)

        # 加入三个区域到主布局
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.list_widget)
        main_layout.addLayout(right_layout)

        self.shapes = []

    def add_shape_from_list(self, item):
        shape_type = self.shape_map.get(item.text(), "triangle")
        color = {
            "triangle": QColor("#e07b39"),    # 橙色三角形
            "diamond": QColor("#c2356b"),     # 紫红菱形
            "trapezoid": QColor("#1c5d4a"),   # 墨绿色梯形
            "hexagon": QColor("#f08c1d")      # 橙黄色六边形
        }.get(shape_type, Qt.black)
        shape = DraggableShape(shape_type, color)
        shape.setPos(100, 100)
        self.scene.addItem(shape)
        self.shapes.append(shape)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene.selectedItems():
                self.scene.removeItem(item)
                if item in self.shapes:
                    self.shapes.remove(item)
            event.accept()
        else:
            super().keyPressEvent(event)

    def send_layout(self):
        shapes_data = []
        for item in self.shapes:
            if item.shape_type == "triangle":
                angle = item.angle % 120
            elif item.shape_type == "diamond":
                angle = item.angle % 180
            elif item.shape_type == "hexagon":
                angle = item.angle % 60
            elif item.shape_type == "trapezoid":
                angle = item.angle
            shape_data = {
                "shape": item.shape_type,
                "x": int(item.x()),
                "y": int(item.y()),
                "angle": round(angle, 1),
            }
            shapes_data.append(shape_data)

        json_data = json.dumps(shapes_data)
        print("发送数据:", json_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ShapeDesigner()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())