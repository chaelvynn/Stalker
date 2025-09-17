import cv2
import face_recognition
import os
from tkinter import *
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageTk

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
        self.tracker = DeepSort(max_age=5)

        self.known_face_encodings, self.known_face_names = load_known_faces("faces")
        self.target_encoding = None
        self.selected_name = "Disable"
        self.validated_track_id = None
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names, command=self.update_tracking_status)
        self.face_menu.pack()

        self.is_tracking = False
        self.prev_tick = cv2.getTickCount()
        self.curr_fps = 0.0
        self.prev_positions = {}

        self.process_frame()

    def update_tracking_status(self, selected_name):
        if selected_name == "Disable":
            self.is_tracking = False
            self.target_encoding = None
            self.validated_track_id = None
            print("Tracking Disabled.")
        else:
            print(f"Tracking enabled for: {selected_name}")
            self.is_tracking = True
            self.selected_name = selected_name
            idx = self.known_face_names.index(selected_name)
            self.target_encoding = self.known_face_encodings[idx]
            self.validated_track_id = None  # reset saat mulai ulang

    def process_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return

        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick

        cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        detections = []

        if self.is_tracking:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in haar_faces:
                # Crop dan resize area wajah
                face_img = frame[y:y+h, x:x+w]
                rgb_face = cv2.resize(face_img, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]

                # Hanya lakukan face recognition jika belum tervalidasi
                if self.validated_track_id is None:
                    encoding = face_recognition.face_encodings(rgb_face)
                    if encoding:
                        distance = face_recognition.face_distance([self.target_encoding], encoding[0])[0]
                        if distance < 0.4:
                            print("Target matched!")
                            detections.append(([x, y, w, h], 0.99, 'face'))  # valid match

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"Validating {self.selected_name}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



                else:
                    detections.append(([x, y, w, h], 0.8, 'face'))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        movement_threshold = 10

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            # Jika baru tervalidasi, simpan track_id
            if self.validated_track_id is None:
                self.validated_track_id = track_id
                print(f"Tracking started. ID: {track_id}")

            # Hanya gambar bbox jika ID cocok
            if track_id == self.validated_track_id:
                x_center = int((l + r) / 2)
                y_center = int((t + b) / 2)
                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.selected_name} | ID {track_id}", (int(l), int(t) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

            if self.validated_track_id is not None:
                valid_ids = [track.track_id for track in tracks if track.is_confirmed()]
                if self.validated_track_id not in valid_ids:
                    print("Target keluar frame. Resetting tracking...")
                    self.validated_track_id = None
                    self.initial_detection_done = False


        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 500))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)
        self.root.after(10, self.process_frame)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceTrackerApp()
    app.run()
