import pygame
import time

def scale_axis(val, deadzone=0.2):
    """
    Ubah axis [-1,1] ke [-100,100] dengan deadzone biar gak sensitif
    """
    if abs(val) < deadzone:
        return 0
    return int(val * 100)

def main():
    # Init pygame & joystick
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick:", joystick.get_name())

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Tombol-tombol
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 7:   # Start
                    print("[BTN] TAKEOFF")
                elif event.button == 6: # Select
                    print("[BTN] LAND")
                elif event.button == 0: # X
                    print("[BTN] EMERGENCY STOP!")

            if event.type == pygame.JOYHATMOTION:
                if event.value[1] == 1:
                    print("[HAT] Speed UP")
                elif event.value[1] == -1:
                    print("[HAT] Speed DOWN")

        # === Axis Control (Mapping drone) ===
        # Stick kiri
        yaw = scale_axis(joystick.get_axis(0))       # rotasi kiri/kanan
        throttle = -scale_axis(joystick.get_axis(1)) # naik/turun

        # Stick kanan
        roll = scale_axis(joystick.get_axis(2))      # geser kiri/kanan
        pitch = -scale_axis(joystick.get_axis(3))    # maju/mundur

        # Cetak ke terminal
        if yaw != 0:
            print(f"[AXIS] Yaw {'kanan' if yaw > 0 else 'kiri'} ({yaw})")
        if throttle != 0:
            print(f"[AXIS] {'Naik' if throttle > 0 else 'Turun'} ({throttle})")
        if roll != 0:
            print(f"[AXIS] Geser {'kanan' if roll > 0 else 'kiri'} ({roll})")
        if pitch != 0:
            print(f"[AXIS] {'Maju' if pitch > 0 else 'Mundur'} ({pitch})")

        time.sleep(0.1) 

    pygame.quit()

if __name__ == "__main__":
    main()
