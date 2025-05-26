import cv2
from driver.HKcamera import Camera
import os
import time


def main():
    camera = Camera(0)
    if not os.path.exists('./imgs'):
        os.makedirs('./imgs')

    while True:
        img = camera.get_img()
        if img is None:
            continue
        resized_img = cv2.resize(img, (1280, 720))  # Resize to fit the screen
        cv2.imshow('Camera', resized_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'./imgs/{timestamp}.jpg'
            cv2.imwrite(filename, img)
            print(f"Image saved to {filename}")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
