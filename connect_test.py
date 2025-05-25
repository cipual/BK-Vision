import socket
import threading
import time

HOST = '192.168.0.10'  # 本机 IP 地址
PORT = 2000             # 监听端口
BUFFER_SIZE = 1024

# 工件信息列表（示例）
workpiece_info_list = [
    "shape=triangle;angle=10.0;x=100;y=200;color=red;",
    "shape=hexagon;angle=45.5;x=150;y=250;color=green;",
    # 添加更多工件信息
]
workpiece_index = 0
area_num = 1
movement_flag = False

def handle_client(conn, addr):
    global workpiece_index, movement_flag, area_num

    print(f"[连接] 来自 {addr}")

    while True:
        try:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                print("[断开连接]")
                break

            msg = data.decode('utf-8').strip()
            print(f"[接收] {msg}")

            if msg == "Start":
                print("[操作] 启动检测流程")
                # 你可以在这里调用拍照、检测函数
                workpiece_index = len(workpiece_info_list)
                movement_flag = False
                area_num += 1

            elif msg == "Sort":
                print("[操作] 重新检测/追踪")
                # 检测或追踪工件
                if not movement_flag:
                    # MovementDetection()
                    pass
                else:
                    # Re_Workpiece_Detection()
                    movement_flag = False

            elif msg == "Stop":
                print("[操作] 停止检测流程")
                # 停止相机或资源释放操作

            elif msg == "OK":
                if workpiece_index > 0:
                    info_to_send = workpiece_info_list[len(workpiece_info_list) - workpiece_index]
                    print(f"[发送] {info_to_send}")
                    conn.sendall(info_to_send.encode('utf-8'))
                    workpiece_index -= 1

        except Exception as e:
            print(f"[异常] {e}")
            break

    conn.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[监听] 等待来自 PLC 的连接 {HOST}:{PORT}...")

    while True:
        conn, addr = server.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.daemon = True
        client_thread.start()

if __name__ == "__main__":
    start_server()
