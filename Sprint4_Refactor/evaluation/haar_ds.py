import cv2
import face_recognition
import os
from tkinter import *
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageTk

def compute_iou(boxA, boxB):
    """
    boxA, boxB: (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0

    return interArea / union

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

        self.state = "SEARCH"   # SEARCH | TRACK
        self.target_bbox = None
        self.validated_track_id = None


        # Metrik
        self.fps_log = []
        self.total_frames = 0
        self.success_frames = 0
        self.prev_point = None
        self.point_motion = []

        # ===== Evaluation =====
        self.evaluation_active = False
        self.eval_start_time = None

        self.prev_center = None
        self.center_displacements = []



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

        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

        self.process_frame()

    def update_tracking_status(self, selected_name):
        if selected_name == "Disable":
            self.is_tracking = False
            self.target_encoding = None
            self.validated_track_id = None
            self.evaluation_active = False
            print("Tracking Disabled.")
        else:
            print(f"Tracking enabled for: {selected_name}")
            self.is_tracking = True
            self.selected_name = selected_name
            idx = self.known_face_names.index(selected_name)
            self.target_encoding = self.known_face_encodings[idx]
            self.validated_track_id = None  # reset saat mulai ulang

            # ===== start evaluation =====
            self.fps_log.clear()
            self.total_frames = 0
            self.success_frames = 0
            self.point_motion.clear()
            self.prev_point = None
            self.evaluation_active = True
            self.eval_start_time = cv2.getTickCount()

            self.state = "SEARCH"
            self.validated_track_id = None

    def process_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return

        # 1. FPS
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick

        if self.evaluation_active:
            self.fps_log.append(self.curr_fps)

        detections = []

        # =========================
        # 2. STATE LOGIC (DI SINI)
        if self.state == "SEARCH" and self.target_encoding is not None:
            rgb_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                dist = face_recognition.face_distance([self.target_encoding], encoding)[0]
                if dist < 0.45:
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    self.target_bbox = (left, top, right-left, bottom-top)
                    print("Target found. Switching to TRACK.")
                    self.state = "TRACK"
                    break


        elif self.state == "TRACK":

            if self.evaluation_active:
                self.total_frames += 1

            tracked_this_frame = False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 5)

            detections = []
            for (x, y, w, h) in faces:
                detections.append(([x, y, w, h], 0.9, "face"))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                l, t, r, b = track.to_ltrb()
                track_id = track.track_id

                # FIRST FRAME: lock ID based on IOU with target_bbox
                if self.validated_track_id is None:
                    iou = compute_iou(self.target_bbox, (l, t, r-l, b-t))
                    if iou > 0.3:
                        self.validated_track_id = track_id
                        print(f"Locked target ID: {track_id}")

                if track_id == self.validated_track_id:
                    tracked_this_frame = True

                    cx = (l + r) / 2
                    cy = (t + b) / 2

                    if self.prev_center is not None:
                        dx = cx - self.prev_center[0]
                        dy = cy - self.prev_center[1]
                        disp = (dx**2 + dy**2) ** 0.5
                        self.center_displacements.append(disp)

                    self.prev_center = (cx, cy)


                    cv2.rectangle(frame, (int(l),int(t)), (int(r),int(b)), (0,255,0), 2)
    
            if self.evaluation_active and tracked_this_frame:
                self.success_frames += 1

            if self.validated_track_id is not None:
                active_ids = [t.track_id for t in tracks if t.is_confirmed()]
                if self.validated_track_id not in active_ids:
                    print("Target lost. Returning to SEARCH.")
                    self.prev_center = None
                    self.state = "SEARCH"
                    self.validated_track_id = None
                    self.target_bbox = None


        # =========================

        # 3. Drawing & command logic
        cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 4. Render Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 500))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)

        self.root.after(10, self.process_frame)

    def cleanup(self):
        print("\n=== Haar + DeepSORT Evaluation ===")

        if len(self.fps_log) > 0:
            print(f"Avg FPS              : {sum(self.fps_log)/len(self.fps_log):.2f}")
        else:
            print("Avg FPS              : N/A")

        if self.total_frames > 0:
            print(f"Tracking Success (%) : {self.success_frames/self.total_frames*100:.2f}")
        else:
            print("Tracking Success (%) : N/A")

        if self.eval_start_time is not None:
            end_time = cv2.getTickCount()
            duration = (end_time - self.eval_start_time) / cv2.getTickFrequency()
            print(f"Tracking Duration (s): {duration:.2f}")
        else:
            print("Tracking Duration (s): N/A")

        if len(self.center_displacements) > 0:
            avg_stability = sum(self.center_displacements) / len(self.center_displacements)
            print(f"Tracking Stability (px) : {avg_stability:.2f}")
        else:
            print("Tracking Stability (px) : N/A")


        self.video.release()
        self.root.destroy()


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceTrackerApp()
    app.run()
