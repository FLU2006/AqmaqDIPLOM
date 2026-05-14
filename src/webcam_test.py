import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-friendly
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    print("Webcam opened. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: cannot read frame")
            break

        cv2.imshow("webcam_test", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()