import cv2
from driver.HKcamera import Camera


def main():
    camera = Camera(0)
    while True:
        img = camera.get_img()
        if img is None:
            continue
        cv2.imshow('Detected', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
