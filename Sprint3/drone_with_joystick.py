import pygame
import cv2
import time
import numpy as np
from djitellopy import tello
from flight_commands import fly

def scale_axis(val):
    """Convert axis value [-1,1] to [-100,100] integer"""
    return int(val * 100)

def main():
    drone = tello.Tello()
    drone.connect()
    print("Battery:", drone.get_battery(), "%")

    drone.streamon()
    frame_read = drone.get_frame_read()

    # Inisialisasi joystick
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print("Joystick:", joystick.get_name())
    speed = 30
    is_flying = False
    is_landing = False

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.JOYBUTTONDOWN:
                    # Button START to take off
                    if event.button == 7 and not is_flying:
                        print("Takeoff")
                        drone.takeoff()
                        is_flying = True
                        is_landing = False
                    
                    # Button SELECT to land
                    elif event.button == 6 and is_flying and not is_landing:
                        print("Landing")
                        drone.land()
                        is_landing = True
                        is_flying = False

                    # Button X to emergency stop the drone
                    elif event.button == 0:
                        print("EMERGENCY STOP!")
                        drone.emergency()
                        is_flying = False
                        is_landing = False

                # set drone speed using D-pad (hat)
                if event.type == pygame.JOYHATMOTION:
                    if event.value[1] == 1:  # Up
                        speed = min(100, speed + 10)
                        drone.set_speed(speed)
                        print("Speed naik:", speed)
                    elif event.value[1] == -1:  # Down
                        speed = max(10, speed - 10)
                        drone.set_speed(speed)
                        print("Speed turun:", speed)

            # ======== Axis Control =========
            # Right axis:
            roll     = scale_axis(joystick.get_axis(2))    # left/right (X)
            pitch    = -scale_axis(joystick.get_axis(3))   # forward/ backward (Z)
            # Left axis:       
            throttle = -scale_axis(joystick.get_axis(1))   # up/down (Y)
            yaw      = scale_axis(joystick.get_axis(0))    # yaw (turn left/ turn right)

            fly([roll, pitch, throttle, yaw], drone)



            frame = frame_read.frame
            if frame is not None:
                frame = cv2.resize(frame, (960, 720))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Tello POV", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            time.sleep(0.05) 




    except Exception as e:
        print("Error:", e)
    finally:
        print("Cleaning up...")
        if drone.is_flying:
            drone.land()
        drone.streamoff()
        drone.end()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


