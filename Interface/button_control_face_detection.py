# Import necessary libraries
from tkinter import Tk, Label, Button, Frame
import cv2
from PIL import Image, ImageTk
import numpy as np

# Class for controlling webcam via keyboard commands
class WebcamController:
    def __init__(self):
        # Initialize the Tkinter window
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)

        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)

        # Label for displaying video stream
        self.cap_lbl = Label(self.root)

        # Button for activating face detection
        self.face_detection_button = Button(self.root, text="Detect Faces", command=self.toggle_face_detection)

        # Variable to track face detection status
        self.face_detection = False

    # Method to toggle face detection
    def toggle_face_detection(self):
        self.face_detection = not self.face_detection

    # Method to run the application
    def run_app(self):
        try:
            # Add button and video stream label to the window
            self.face_detection_button.grid(column=0, row=2)
            self.cap_lbl.grid(column=1, row=1)

            # Call the video stream method
            self.video_stream()

            # Start the tkinter main loop
            self.root.mainloop()

        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            # Clean up resources
            self.cleanup()

    # Method to display video stream
    def video_stream(self):
        # Read a frame from the webcam
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check if face detection is enabled
            if self.face_detection:
                self.detect_faces(frame_rgb)

            # Convert frame to ImageTk format
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the frame
            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)

        # Update the video stream label with the current frame
        self.cap_lbl.after(5, self.video_stream)

    # Method for face detection
    def detect_faces(self, frame):
        """Print the number of faces detected in the drones camera stream to the output window and
        draws a box around any faces in the cv2 window if they are detected"""

        # Create a cascade
        faceCascade = cv2.CascadeClassifier("faceDetectionfr/data-files/haarcascade_frontalface_default.xml")

        # Covert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get numpy array with values for faces detected by passing in grayscale image, scale factor, and minimum neighbors
        faces = faceCascade.detectMultiScale(frame_gray, 1.2, 3)

        # For the x, y coordinates and width, height detected
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face using these values
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Update the face count with the number of faces detected
        face_count = "Faces: " + str(len(faces))
        cv2.putText(frame, face_count, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1,
                    2)  # Print the face count as standard output
        print(face_count)


    # Method for cleaning up resources
    def cleanup(self):
        try:
            # Release webcam
            print("Cleaning up resources...")
            self.cap.release()
            self.root.quit()  # Quit the Tkinter main loop
        except Exception as e:
            print(f"Error performing cleanup: {e}")


if __name__ == "__main__":
    # Initialize the GUI
    gui = WebcamController()
    # Run the application
    gui.run_app()
