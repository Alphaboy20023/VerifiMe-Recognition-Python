# python main.py
from app.camera import Camera;
import cv2;

def main():
    cam = Camera()  # use default camera
    try:
        cam.open()  # open camera
        print("Camera opened. Press 'q' to quit.")

        for frame in cam.stream():
            cv2.imshow("Camera Stream", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        cam.close()
        cv2.destroyAllWindows(), # closes all OpenCV windows that were opened
        print("Camera closed.")

if __name__ == "__main__":
    main()
