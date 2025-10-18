from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
import cv2, face_recognition, os, pygame, time
from PIL import Image, ImageTk
from djitellopy import tello
from flight_commands import start_flying, stop_flying
import numpy as np
import threading as thread
import mediapipe as mp

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(directory):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

# Function to calculate the similarity between face encoding and face in the frame
def calculate_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold: #kalau ga mirip 0%
        linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
        return max(0.0, min(1.0, linear_val)) * 100
    else: #kalau mirip 100%/
        linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
        return max(0.0, min(1.0, linear_val)) * 100

# Function to perform face recognition on a single frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    
    face_names = []
    face_confidences = []
    face_distances = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0.0
        distance = 0.0

        face_distances_current = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances_current)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = calculate_confidence(face_distances_current[best_match_index])
            distance = face_distances_current[best_match_index]  # Set distance
 
        face_names.append(name)
        face_confidences.append(confidence)
        face_distances.append(distance)

    return face_locations, face_names, face_confidences, face_distances


class DroneController:
    def __init__(self):
        
        self.root = Tk()
        self.root.title("Drone Keyboard Controller - Tkinter")
        self.root.minsize(800, 600)

        
        self.input_frame = Frame(self.root)

        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()
       
        self.cap = self.drone.get_frame_read()

        print("Battery: ", self.drone.get_battery(), "%")
        self.drone.speed = 100
        

        #joystick initialization
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Joystick:", self.joystick.get_name())

        self.joystick_running = True
        
        self.is_flying = False
        self.is_landing = False


        faces_dir = "faces" #directory
        self.known_face_encodings, self.known_face_names = load_known_faces(faces_dir)

        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.demo_button = Button(self.button_frame, text="Takeoff/Land", command=self.takeoff_land)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names)

        # Camera and face parameters
        self.FOCAL_LENGTH = 800  # Adjust this value based on your camera
        self.KNOWN_FACE_WIDTH = 16  # Average width of a human face in cm

        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_name = "Disable"
        self.old_distance = 0.0
        self.isPositionRight = 2
        self.prev_move = ""
        self.movement = []
        self.is5thframe = 0

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        
        self.lk_params = dict(
                    winSize=(25, 25),  
                    maxLevel=4,  
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

       
        
    
    def takeoff_land(self):
        
        if self.drone.is_flying:
            thread.Thread(target=self.drone.land).start()
        else:
            thread.Thread(target=self.drone.takeoff).start()

    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels
    

    
    def joystick_control(self):
        def scale_axis(value, deadzone = 0.1):
            if abs(value) < deadzone:
                return 0
            # Change joystick value from [-1, 1] to [-100, 100]
            return int(value * 100)
        
        last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0

        while self.joystick_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.JOYBUTTONDOWN:
                    # Button START to take off
                    if event.button == 7 and not self.is_flying:
                        print("Takeoff")
                        self.drone.takeoff()
                        self.is_flying = True
                        self.is_landing = False
                    
                    # Button SELECT to land
                    elif event.button == 6 and self.is_flying and not self.is_landing:
                        print("Landing")
                        self.drone.land()
                        self.is_landing = True
                        self.is_flying = False

                    # Button X to emergency stop the drone
                    elif event.button == 0:
                        print("EMERGENCY STOP!")
                        self.drone.emergency()
                        self.is_flying = False
                        self.is_landing = False

                # set drone speed using D-pad (hat)
                if event.type == pygame.JOYHATMOTION:
                    if event.value[1] == 1:  # Up
                        self.drone.speed = min(100, self.drone.speed + 10)
                        print("Speed naik:", self.drone.speed)
                    elif event.value[1] == -1:  # Down
                        self.drone.speed = max(10, self.drone.speed - 10)
                        print("Speed turun:", self.drone.speed)
                
            #Axis control
            # Right axis:
            roll     = scale_axis(self.joystick.get_axis(2))    # left/right (X)
            pitch    = -scale_axis(self.joystick.get_axis(3))   # forward/ backward (Z)
            # Left axis:       
            throttle = -scale_axis(self.joystick.get_axis(1))   # up/down (Y)
            yaw      = scale_axis(self.joystick.get_axis(0))    # yaw (turn left/ turn right)



            # Print commands
            if roll!= 0 or pitch !=0 or throttle != 0 or yaw != 0:

                self.drone.send_rc_control(roll, pitch, throttle, yaw)
                last_roll, last_pitch, last_throttle, last_yaw = roll, pitch, throttle, yaw
                if roll != 0:
                    if roll > 0:
                        # self.movement.append("[C]Move right")
                        print("[C]Moving Right:", roll)
                    elif roll < 0:
                        # self.movement.append("[C]Move left")
                        print("[C]Moving Left:", roll)
                if pitch != 0:
                    if pitch > 0:
                        # self.movement.append("[C]Move forward")
                        print("[C]Moving Forward:", pitch)
                    elif roll < 0:
                        # self.movement.append("[C]Move backward")
                        print("[C]Moving Backward:", pitch)
                if throttle != 0:
                    if throttle > 0:
                        # self.movement.append("[C]Move upward")
                        print("[C]Moving Up:", throttle)
                    elif throttle < 0:
                        # self.movement.append("[C]Move downward")
                        print("[C]Moving Down:", throttle)
                if yaw != 0:
                    if yaw > 0:
                        # self.movement.append("[C]Yaw right")
                        print("[C]Yaw Right:", yaw)
                    elif yaw < 0:
                        # self.movement.append("[C]Yaw left")
                        print("[C]Yaw Left:", yaw)

            elif roll == pitch == throttle == yaw == 0 and (last_roll != 0 or last_pitch != 0 or last_throttle != 0 or last_yaw != 0):
                self.drone.send_rc_control(0, 0, 0, 0)
                last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0
                # print("Stopping movement")

            time.sleep(0.1)
            


    def run_app(self):
        try:
            """CONTROL BY KEYBOARD"""
            # Bind the key presses with to the flight commands by associating them with a direction to travel.
            self.input_frame.bind('<KeyPress-w>', lambda event: start_flying(event, 'forward', self.drone, 100))
            self.input_frame.bind('<KeyRelease-w>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-a>', lambda event: start_flying(event, 'left', self.drone, 100))
            self.input_frame.bind('<KeyRelease-a>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-s>', lambda event: start_flying(event, 'backward', self.drone, 100))
            self.input_frame.bind('<KeyRelease-s>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-d>', lambda event: start_flying(event, 'right', self.drone, 100))
            self.input_frame.bind('<KeyRelease-d>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Up>', lambda event: start_flying(event, 'upward', self.drone, 100))
            self.input_frame.bind('<KeyRelease-Up>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Down>', lambda event: start_flying(event, 'downward', self.drone, 100))
            self.input_frame.bind('<KeyRelease-Down>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-q>', lambda event: start_flying(event, 'yaw_left', self.drone, 100))
            self.input_frame.bind('<KeyRelease-Left>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-e>', lambda event: start_flying(event, 'yaw_right', self.drone, 100))
            self.input_frame.bind('<KeyRelease-Right>', lambda event: stop_flying(event, self.drone))
            
            
            
            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)
            self.video_stream()
 
            self.root.mainloop()

           

        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            self.cleanup()
            print("Cleaning up...")
            if self.drone.is_flying:
                self.drone.land()
            self.drone.streamoff()
            self.drone.end()
            pygame.quit()

  
    def dummy_function(self, event, key):
        print(f"Key {key} pressed")

    
    def video_stream(self):
        h, w = 480, 720
        
        frame =  None

        if self.prev_frame_gray is None:
            frame = self.cap.frame
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        def read_frame():
            nonlocal frame
            frame = self.cap.frame
            
        

        def read_movement():
            if self.movement:
                for movement in self.movement:
                    if self.isPositionRight == 0:
                        if self.prev_move == "Move forward":
                            start_flying(None, 'backward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move backward":
                            start_flying(None, 'forward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move left":
                            start_flying(None, 'right', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move right":
                            start_flying(None, 'left', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move upward":
                            start_flying(None, 'downward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move downward":
                            start_flying(None, 'upward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        self.prev_move = ""
                        break

                    if movement == "Move forward1":
                        start_flying(None, 'forward', self.drone, 20)
                    if movement == "Move forward2":
                        start_flying(None, 'forward', self.drone, 30)    
                    elif movement == "Move backward1":
                        start_flying(None, 'backward', self.drone, 20)
                    elif movement == "Move backward2":
                        start_flying(None, 'backward', self.drone, 30)    
                    elif movement == "Move left1":
                        start_flying(None, 'left', self.drone, 20)
                    elif movement == "Move left2":
                        start_flying(None, 'left', self.drone, 30)
                    elif movement == "Move right1":
                        start_flying(None, 'right', self.drone, 20)
                    elif movement == "Move right2":
                        start_flying(None, 'right', self.drone, 30)
                    elif movement == "Move upward":
                        start_flying(None, 'upward', self.drone, 30)
                    elif movement == "Move downward":
                        start_flying(None, 'downward', self.drone, 30)
                    self.prev_move = movement
                self.movement.clear()


        def adjust_color_balance(frame):
            
            b, g, r = cv2.split(frame)
            
            
            r = cv2.add(r, 70) 
            g = cv2.add(g, 20)  
            
            # Merge back the channels
            frame = cv2.merge([b, g, r])
            
            return frame
        
        t1 = thread.Thread(target=read_frame)
        t1.start()
        t1.join()

        #PREPROCESS
        if frame is not None:
            
            frame = cv2.resize(frame, (w, h))
            
            # denoised_image = cv2.medianBlur(frame, 5) 
            # sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # sharpened_image = cv2.filter2D(denoised_image, -1, sharpening_kernel)
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            selected_name = self.face_detection_var.get()

            thread.Thread(target=self.joystick_control, daemon=True).start()

            if selected_name == "Disable":
                self.joystick_running = True
            else:
                self.joystick_running = False

            if selected_name != self.current_name:
                self.current_name = selected_name
                self.tracked_points = None
                self.movement.clear()
            
            if selected_name == self.current_name:
                t3 = thread.Thread(target=read_movement)
                t3.start()
                t3.join()
               

                if self.tracked_points is None:
                    
                    self.current_name = selected_name
                    if selected_name != "Disable":
                        if selected_name in self.known_face_names:
                            idx = self.known_face_names.index(selected_name)
                            target_encodings = [self.known_face_encodings[idx]]
                            target_names = [self.known_face_names[idx]]
                        else:
                            target_encodings = self.known_face_encodings
                            target_names = self.known_face_names

                        
                        face_locations, face_names, face_confidences, face_distances = [], [], [], []
                        def recognize_faces_thread():
                            nonlocal face_locations, face_names, face_confidences, face_distances
                            face_locations, face_names, face_confidences, face_distances = recognize_faces(frame, target_encodings, target_names)
                        
                        t2 = thread.Thread(target=recognize_faces_thread)
                        t2.start()
                        t2.join()

                        for (top, right, bottom, left), name, confidence, distance in zip(face_locations, face_names, face_confidences, face_distances):
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            
                            if name != "Unknown":
                                self.movement.clear()
                                face_width_pixels = right - left
                                distance = self.calculate_distance(face_width_pixels) 
                                self.old_distance = distance
                                self.tracked_points = np.array([[[(left + right) // 2, (top + bottom) // 2]]], dtype=np.float32)
                                x, y = self.tracked_points[0].ravel()
                                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                                
                                if distance > 85 and distance < 95:
                                    self.movement.append("Move forward1")
                                    self.isPositionRight = 1
                                elif distance > 95:
                                    self.movement.append("Move forward2")
                                    self.isPositionRight = 1
                                elif distance < 60 and distance > 50:
                                    self.movement.append("Move backward1")
                                    self.isPositionRight = 1
                                elif distance < 60 and distance > 50:
                                    self.movement.append("Move backward2")
                                    self.isPositionRight = 1

                                
                                face_center_x = (left + right) // 2
                                frame_center_x = w // 2
                                if face_center_x < frame_center_x - 120:
                                    self.movement.append("Move left")
                                    self.isPositionRight = 1
                                elif face_center_x > frame_center_x + 120:
                                    self.movement.append("Move right")
                                    self.isPositionRight = 1

                               
                                face_center_y = (top + bottom) // 2
                                frame_center_y = h // 2
                                if face_center_y < frame_center_y - 90:
                                    self.movement.append("Move upward")
                                    self.isPositionRight = 1
                                elif face_center_y > frame_center_y + 90:
                                    self.movement.append("Move downward")
                                    self.isPositionRight = 1
                                
                                if self.isPositionRight == 0:
                                    self.movement.clear()

                                if self.isPositionRight == 1:
                                    self.isPositionRight = 2
                            
                                if self.isPositionRight == 2:
                                    self.isPositionRight = 0
                                    

                                if self.isPositionRight == 0:
                                    self.isPositionRight = 2
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
                elif self.tracked_points is not None:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_frame_gray, cur_gray_frame, self.tracked_points, None, **self.lk_params
                    )

                    self.tracked_points = next_points
        
                    if next_points is not None and len(next_points) > 0:
                        x, y = next_points[0].ravel()
                        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                   
                        faces = []
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                h, w, _ = frame.shape
                                xmin, ymin = int(bboxC.xmin * w), int(bboxC.ymin * h)
                                width, height = int(bboxC.width * w), int(bboxC.height * h)
                                faces.append((xmin, ymin, xmin + width, ymin + height))

                        matched_face = None

                        for face in faces:
                            left, top, right, bottom = face
                            if left <= x <= right and top <= y <= bottom:
                                matched_face = face
                                break        
                        
                        if matched_face:
                            face_width = matched_face[2] - matched_face[0]
                            distance = self.calculate_distance(face_width)
                            print(f"Estimated Distance: {distance:.2f} cm")
                            self.movement.clear()
                            if distance > 85 and distance < 95:
                                self.movement.append("Move forward1")
                                self.isPositionRight = 1
                            elif distance > 95:
                                self.movement.append("Move forward2")
                                self.isPositionRight = 1
                            elif distance < 60 and distance > 50:
                                self.movement.append("Move backward1")
                                self.isPositionRight = 1
                            elif distance < 60 and distance > 50:
                                self.movement.append("Move backward2")
                                self.isPositionRight = 1

                            
                            face_center_x = (left + right) // 2
                            frame_center_x = w // 2
                            if x < frame_center_x - 120 and x > frame_center_x - 150:
                                self.movement.append("Move left1")
                                self.isPositionRight = 1
                            elif x < frame_center_x - 150:
                                self.movement.append("Move left2")
                                self.isPositionRight = 1
                            elif face_center_x > frame_center_x + 120 and x < frame_center_x + 150:
                                self.movement.append("Move right1")
                                self.isPositionRight = 1
                            elif face_center_x > frame_center_x + 150:
                                self.movement.append("Move right2")
                                self.isPositionRight = 1

                           
                            face_center_y = (top + bottom) // 2
                            frame_center_y = h // 2
                            if y < frame_center_y - 90:
                                self.movement.append("Move upward")
                                self.isPositionRight = 1
                            elif face_center_y > frame_center_y + 90:
                                self.movement.append("Move downward")
                                self.isPositionRight = 1

                            if self.isPositionRight == 1:
                                self.isPositionRight = 2
                            
                            if self.isPositionRight == 2:
                                self.isPositionRight = 0
                                

                            if self.isPositionRight == 0:
                                self.isPositionRight = 2  
                        else:
                            print("Tracked point does not match any face. Resetting tracking.")
                            self.tracked_points = None
                            self.movement.clear()
                            
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        else:
                           
                            print("Tracked point out of bounds. Resetting tracking.")
                            self.tracked_points = None
                            self.movement.clear()
                else:
                  
                    print("Optical flow returned invalid points. Resetting tracking.")
                    self.tracked_points = None
                    self.movement.clear()



            self.prev_frame_gray = cur_gray_frame.copy()
            # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)

        self.cap_lbl.after(10, self.video_stream)

    def cleanup(self) -> None:
        try:
           
            print("Cleaning up resources...")
            self.drone.end()
            self.root.quit()  
            exit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    gui = DroneController()
    gui.run_app()
