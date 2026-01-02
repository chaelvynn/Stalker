import cv2
import face_recognition
import os, numpy as np
from tkinter import *
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
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

        self.state = "SEARCH"
        self.target_bbox = None
        self.validated_track_id = None


        # ===== Evaluation Metrics =====
        self.evaluation_active = False

        self.fps_log = []
        self.total_frames = 0
        self.success_frames = 0

        self.id_switch_count = 0
        self.last_track_id = None

        self.bbox_motion = []
        self.prev_bbox_center = None


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

        self.eval_start_time = None

        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

        self.process_frame()

    def start_evaluation(self):
        self.fps_log.clear()
        self.total_frames = 0
        self.success_frames = 0
        self.id_switch_count = 0
        self.last_track_id = None
        self.bbox_motion.clear()
        self.prev_bbox_center = None
        self.evaluation_active = True
        print("Evaluation started.")


    def update_tracking_status(self, selected_name):
        self.state = "SEARCH"
        self.validated_track_id = None
        self.target_bbox = None

        if selected_name == "Disable":
            self.start_evaluation()
            self.is_tracking = False
            self.target_encoding = None
            print("Tracking Disabled.")
        else:
            print(f"Tracking enabled for: {selected_name}")
            self.start_evaluation()
            self.eval_start_time = cv2.getTickCount()
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

        if self.evaluation_active:
            self.fps_log.append(self.curr_fps)


        detections = []

        rgb_frame = frame[:, :, ::-1]

        if self.state == "SEARCH" and self.target_encoding is not None:
            rgb_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                dist = face_recognition.face_distance([self.target_encoding], encoding)[0]
                if dist < 0.45:
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    self.target_bbox = (left, top, right-left, bottom-top)
                    self.state = "TRACK"
                    print("Target found → TRACK")
                    break

        elif self.state == "TRACK":
            detections = []

            results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    detections.append(([x, y, bw, bh], det.score[0], "face"))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            #evaluation
            if self.evaluation_active:
                self.total_frames += 1

            tracked_this_frame = False

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                l, t, r, b = track.to_ltrb()
                track_id = track.track_id

                if self.validated_track_id is None:
                    iou = compute_iou(self.target_bbox, (l, t, r-l, b-t))
                    if iou > 0.3:
                        self.validated_track_id = track_id
                        print(f"Locked ID: {track_id}")

                if track_id == self.validated_track_id:

                    tracked_this_frame = True #evaluation
                    if self.last_track_id is not None and track_id != self.last_track_id: #evaluation
                        self.id_switch_count += 1 #evaluation
                    self.last_track_id = track_id #evaluation


                    cx = (l + r) / 2
                    cy = (t + b) / 2

                    if self.prev_bbox_center is not None:
                        # dx = abs(cx - self.prev_bbox_center[0])
                        # dy = abs(cy - self.prev_bbox_center[1])
                        dist = np.linalg.norm([cx - self.prev_bbox_center[0], cy - self.prev_bbox_center[1]]) #calculate euclidean distance
                        
                        self.bbox_motion.append(dist)

                    self.prev_bbox_center = (cx, cy)

                    cv2.rectangle(frame, (int(l),int(t)), (int(r),int(b)), (0,255,0), 2)

                if self.validated_track_id is not None:
                    active_ids = [t.track_id for t in tracks if t.is_confirmed()]
                    if self.validated_track_id not in active_ids:
                        print("Target lost → SEARCH")
                        self.state = "SEARCH"
                        self.validated_track_id = None
                        self.target_bbox = None

            if self.evaluation_active and tracked_this_frame:
                self.success_frames += 1




        cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 500))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)

        self.root.after(10, self.process_frame)

    def cleanup(self):
        print("\n=== MP + DeepSORT Evaluation ===")

        if len(self.fps_log) > 0:
            print(f"Avg FPS              : {sum(self.fps_log)/len(self.fps_log):.2f}")
        else:
            print("Avg FPS              : N/A")

        if self.total_frames > 0:
            print(f"Tracking Success (%) : {self.success_frames/self.total_frames*100:.2f}")
        else:
            print("Tracking Success (%) : N/A")

        print(f"Identity Switches    : {self.id_switch_count}")

        if self.bbox_motion:
            print(f"Tracking Stability   : {sum(self.bbox_motion)/len(self.bbox_motion):.2f}")
        else:
            print("Tracking Stability   : N/A")

        if self.eval_start_time is not None:
            end_time = cv2.getTickCount()
            duration = (end_time - self.eval_start_time) / cv2.getTickFrequency()
            print(f"Tracking Duration (s): {duration:.2f}")

        self.video.release()
        self.root.destroy()


    def run(self):
        self.root.mainloop()



if __name__ == "__main__":
    app = FaceTrackerApp()
    app.run()
