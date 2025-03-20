import datetime
import os
import threading

import cv2
import numpy as np
import time
from djitellopy import Tello
import configargparse
from cvfpscalc import CvFpsCalc
from tello_keyboard_controller import TelloKeyboardController
import ffmpeg
import pyautogui

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

#Function that receives and saves Tello's video feed into .mp4 using ffmpeg-python
def videoRecv():
    try:
        (ffmpeg.input('udp://0.0.0.0:11111', t=30) #t is the amount of time video will be saved. 30 seconds. Can increase or decrease amount of time.
        .output('drone.mp4') #Saves to same directory that I am running code from. Can specify directory with a path.
        .run()
        )
    except Exception as e:
        print("Error receiving video: " + str(e))

drone = Tello()  # declaring drone object
time.sleep(2.0)  # waiting 2 seconds
print("Connecting...")
drone.connect()
print("BATTERY: ")
print(drone.get_battery())
time.sleep(1.0)
print("Loading...")
drone.streamon()  # start camera streaming

def main():
    # Argument parsing
    args = get_args()
    KEYBOARD_CONTROL = args.is_keyboard
    WRITE_CONTROL = False
    in_flight = False

    # Init Tello Controllers
    keyboard_controller = TelloKeyboardController(drone)

    def tello_control(key, keyboard_controller):
        keyboard_controller.control(key)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    # set points (center of the frame coordinates in pixels)
    rifX = 960 / 2
    rifY = 720 / 2

    # PI constant
    Kp_X = 0.1
    Ki_X = 0.0
    Kp_Y = 0.2
    Ki_Y = 0.0

    # Loop time
    Tc = 0.05

    # Set up the parameters for object detection
    conf_threshold = 0.5

    # PI terms initialized
    integral_X = 0
    error_X = 0
    previous_error_X = 0
    integral_Y = 0
    error_Y = 0
    previous_error_Y = 0

    centroX_pre = rifX
    centroY_pre = rifY

    # neural network
    net = cv2.dnn.readNetFromCaffe("C:\\Users\\60189\\PycharmProjects\\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone\\MobileNetSSD_deploy.prototxt.txt", "C:\\Users\\60189\\PycharmProjects\\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone\\MobileNetSSD_deploy.caffemodel")  # modify with the NN path
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    while True:
        fps = cv_fps_calc.get()
        start = time.time()
        frame = drone.get_frame_read().frame

        # Process Key (ESC: end)
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            if not in_flight:
                # Take-off drone
                drone.takeoff()
                in_flight = True

            elif in_flight:
                # Land tello
                drone.land()
                in_flight = False

        elif key == ord('k'):
            mode = 0
            KEYBOARD_CONTROL = True
            WRITE_CONTROL = False
            drone.send_rc_control(0, 0, 0, 0)  # Stop moving
        elif key == ord('n'):
            mode = 1
            WRITE_CONTROL = True
            KEYBOARD_CONTROL = True

        if WRITE_CONTROL:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Start control thread
        threading.Thread(target=tello_control, args=(key, keyboard_controller,)).start()

        cv2.circle(frame, (int(rifX), int(rifY)), 1, (0, 0, 255), 10)

        h, w, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame,
                                 0.007843, (180, 180), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            idx = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]

            if (CLASSES[idx] == "car" or CLASSES[idx] == "motorbike") and confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                          colors[idx], 2)
                # draw the center of the car and motorbike detected
                centroX = (startX + endX) / 2
                centroY = (2 * startY + endY) / 3

                centroX_pre = centroX
                centroY_pre = centroY

                cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

                error_X = -(rifX - centroX)
                error_Y = rifY - centroY

                cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

                # PI controller
                integral_X = integral_X + error_X * Tc  # updating integral PID term
                uX = Kp_X * error_X + Ki_X * integral_X  # updating control 42variable uX
                previous_error_X = error_X  # update previous error variable

                integral_Y = integral_Y + error_Y * Tc  # updating integral PID term
                uY = Kp_Y * error_Y + Ki_Y * integral_Y
                previous_error_Y = error_Y

                drone.send_rc_control(0, 0, round(uY), round(uX))
                # break when a car or motorbike is recognized

                break


            else:  # if nobody is recognized take as reference centerX and centerY of the previous frame
                centroX = centroX_pre
                centroY = centroY_pre
                cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

                error_X = -(rifX - centroX)
                error_Y = rifY - centroY

                cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

                integral_X = integral_X + error_X * Tc  # updating integral PID term
                uX = Kp_X * error_X + Ki_X * integral_X  # updating control variable uX
                previous_error_X = error_X  # update previous error variable

                integral_Y = integral_Y + error_Y * Tc  # updating integral PID term
                uY = Kp_Y * error_Y + Ki_Y * integral_Y
                previous_error_Y = error_Y

                drone.send_rc_control(0, 0, round(uY), round(uX))

                continue

        cv2.imshow("Frame", frame)

        end = time.time()
        elapsed = end - start
        if Tc - elapsed > 0:
            time.sleep(Tc - elapsed)
        end_ = time.time()
        elapsed_ = end_ - start
        fps = 1 / elapsed_
        print("FPS: ", fps)

    drone.streamoff()
    cv2.destroyAllWindows()
    #drone.land()
    print("Landing...")
    print("BATTERY: ")
    print(drone.get_battery())
    drone.end()

if __name__ == '__main__':
    main()