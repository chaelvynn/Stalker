import cv2
import face_recognition
import os
from tkinter import *
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
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

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names, command=self.update_tracking_status)
        self.face_menu.pack()

        self.is_tracking = False
        self.prev_positions = {}
        self.prev_tick = cv2.getTickCount()
        self.curr_fps = 0.0
        self.initial_detection_done = False

        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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
            self.initial_detection_done = False

    def process_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return

        # FPS calculation
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick

        detections = []

        rgb_frame = frame[:, :, ::-1]

        if self.is_tracking:
            # Resize for faster face_recognition
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                distance = face_recognition.face_distance([self.target_encoding], encoding)[0]
                if distance < 0.5:
                    # Convert coords to full-size
                    left, top, right, bottom = left*2, top*2, right*2, bottom*2
                    face_roi = frame[top:bottom, left:right]

                    results = self.mp_face.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    if results.detections:
                        for det in results.detections:
                            bbox = det.location_data.relative_bounding_box
                            x = int(left + bbox.xmin * (right - left))
                            y = int(top + bbox.ymin * (bottom - top))
                            w = int(bbox.width * (right - left))
                            h = int(bbox.height * (bottom - top))

                            confidence = det.score[0]
                            detections.append(([x, y, w, h], confidence, 'face'))

        tracks = self.tracker.update_tracks(detections, frame=frame)

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
                color = (0, 0, 255)
                bbox_width = r - l
                estimated_distance_cm = (800 * 16) / bbox_width if bbox_width > 0 else 0
                text = f"{self.selected_name} | {int(estimated_distance_cm)}cm"
                self.initial_detection_done = True

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, text, (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.prev_positions[track_id] = (x_center, y_center)

        cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
