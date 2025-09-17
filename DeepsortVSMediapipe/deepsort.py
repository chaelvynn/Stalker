import cv2
from cv2 import getTickCount
import face_recognition
import os
from tkinter import *
from deep_sort_realtime.deepsort_tracker import DeepSort
from imutils.video import FPS



def load_known_faces(faces_dir):
    encodings = []
    names = []

    for filename in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, filename)

        if os.path.isfile(img_path):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(filename)[0])  
    return encodings, names


class FaceTrackerApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Face Tracking GUI")
        self.root.minsize(800, 600)

        self.input_frame = Frame(self.root)
        self.input_frame.pack()

        self.button_frame = Frame(self.root)
        self.button_frame.pack()

        self.cap_lbl = Label(self.root)
        self.cap_lbl.pack()

        self.video = cv2.VideoCapture(0)

        # DeepSORT
        self.tracker = DeepSort(max_age=5)

        self.known_face_encodings, self.known_face_names = load_known_faces("faces")
        self.target_encoding = None
        self.selected_name = "Disable"

        # Dropdown menu
        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names, command=self.update_tracking_status)
        self.face_menu.pack()

        self.initial_detection_done = False
        self.prev_positions = {}  # track_id: (x_center, y_center)


        self.is_tracking = False

        self.prev_tick = cv2.getTickCount()
        self.curr_fps = 0.0

        self.tracking_frame_count = 0
        self.tracking_total_time = 0.0


        self.process_frame()





    def update_tracking_status(self, selected_name):
        if selected_name == "Disable":
            self.is_tracking = False
            self.target_encoding = None
            print("Tracking Disabled.")
        else:
            print(f"Tracking enabled for: {selected_name}")
            self.is_tracking = True
            self.selected_name = selected_name
            idx = self.known_face_names.index(selected_name)
            self.target_encoding = self.known_face_encodings[idx]

    def process_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return
        
        # FPS
        curr_tick = cv2.getTickCount()
        # elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        # self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        # self.prev_tick = curr_tick

        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick

        if self.is_tracking:
            self.tracking_frame_count += 1
            self.tracking_total_time += elapsed


        cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]
        detections = []

        if self.is_tracking:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            FOCAL_LENGTH = 800 
            KNOWN_FACE_WIDTH = 16  #cm

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                distance = face_recognition.face_distance([self.target_encoding], encoding)[0]
                match = distance < 0.5

                if match:
                    left *= 2
                    top *= 2
                    right *= 2
                    bottom *= 2

                    x, y, w, h = left, top, right - left, bottom - top
                    confidence_percent = max(0, min(100, (1 - distance) * 100))

                    if w > 0:
                        estimated_distance_cm = (FOCAL_LENGTH * KNOWN_FACE_WIDTH) / w
                    else:
                        estimated_distance_cm = 0

                    # DeepSORT pakai (x, y, w, h)
                    detections.append(([x, y, w, h], confidence_percent / 100, 'face'))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        movement_threshold = 10


        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            l, t, r, b = track.to_ltrb()
            track_id = track.track_id
            x_center = int((l + r) / 2)
            y_center = int((t + b) / 2)

            color = (0, 255, 0) 
            text = f"ID {track_id}"

            if not self.initial_detection_done:
                # Frame pertama saat wajah dikenali
                color = (0, 0, 255)

                bbox_width = r - l
                estimated_distance_cm = (800 * 16) / bbox_width if bbox_width > 0 else 0

                text = f"{self.selected_name} | {int(confidence_percent)}% | {int(estimated_distance_cm)}cm"
                self.initial_detection_done = True

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, text, (int(l), int(t) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Logging arah gerakan
            prev_pos = self.prev_positions.get(track_id, None)
            if prev_pos:
                dx = x_center - prev_pos[0]
                dy = y_center - prev_pos[1]
                movement = ""

                if abs(dx) > movement_threshold:
                    movement += "Right" if dx > 0 else "Left"
                if abs(dy) > movement_threshold:
                    movement += " Down" if dy > 0 else " Up"

                if movement.strip():
                    print(f"Moved {movement.strip()}")

            self.prev_positions[track_id] = (x_center, y_center)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 500))
        from PIL import Image, ImageTk
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)

        self.root.after(10, self.process_frame)

    def show_avg_fps(self):
        if self.tracking_total_time > 0:
            avg_fps = self.tracking_frame_count / self.tracking_total_time
            print(f"Average FPS during tracking: {avg_fps:.2f}")
        else:
            print("No tracking data available yet.")


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceTrackerApp()
    app.run()
