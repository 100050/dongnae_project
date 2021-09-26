import cv2

frame  = cv2.imread('cctv1.png', cv2.IMREAD_COLOR)

size = (320, 240)
frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
while True:

    key = cv2.waitKey(33)
    if key == 27:
            break

    cv2.imshow("VideoFrame", frame_resized)


