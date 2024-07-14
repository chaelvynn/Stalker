# import Tkinter to create our GUI.
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
# import openCV for receiving the video frames
import cv2
# make imports from the Pillow library for displaying the video stream with Tkinter.
from PIL import Image, ImageTk

# Class for controlling the webcam video stream via keyboard commands
class WebcamController:
    def __init__(self):
        # Initialize the Tkinter window, give it a title, and define its minimum size on the screen.
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)

        # Create a hidden frame to handle input from key presses and releases
        self.input_frame = Frame(self.root)

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

        # Load the face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Label for displaying video stream
        self.cap_lbl = Label(self.root)

        # Frame to hold buttons
        self.button_frame = Frame(self.root)

        # Create a button to demonstrate functionality
        self.demo_button = Button(self.button_frame, text="Demo Button", command=self.demo_function)

        # Variable for dropdown menu
        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")

        # Create a dropdown menu for face detection
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", "Enable")

    # Define a demo function for the button
    def demo_function(self):
        print("Button clicked!")

    # Method to run the application
    def run_app(self):
        try:
            # Pack the hidden frame and give direct input focus to it.
            self.input_frame.pack()
            self.input_frame.focus_set()

            # Pack the video label
            self.cap_lbl.pack(anchor="center", pady=15)

            # Add the buttons to the button frame and pack it below the video
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)

            # Call the video stream method
            self.video_stream()

            # Start the tkinter main loop
            self.root.mainloop()

        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            # When the root window is exited out of ensure to clean up any resources.
            self.cleanup()

    # Dummy function to demonstrate key bindings
    def dummy_function(self, event, key):
        print(f"Key {key} pressed")

    # Method to display video stream
    def video_stream(self):
        # Define the height and width to resize the current frame to
        h = 480
        w = 720

        # Read a frame from the webcam
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (w, h))

            # Convert the current frame to the rgb colorspace
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # Check if face detection is enabled
            if self.face_detection_var.get() == "Enable":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(cv2image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert this to a Pillow Image object
            img = Image.fromarray(cv2image)

            # Convert this then to a Tkinter compatible PhotoImage object
            imgtk = ImageTk.PhotoImage(image=img)

            # Set it to the photo image
            self.cap_lbl.imgtk = imgtk

            # Configure the photo image as the displayed image
            self.cap_lbl.configure(image=imgtk)

        # Update the video stream label with the current frame
        # by recursively calling the method itself with a delay.
        self.cap_lbl.after(10, self.video_stream)

    # Method for cleaning up resources
    def cleanup(self) -> None:
        try:
            # Release any resources
            print("Cleaning up resources...")
            self.cap.release()
            self.root.quit()  # Quit the Tkinter main loop
            exit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    # Initialize the GUI
    gui = WebcamController()
    # Call the run_app method to run tkinter mainloop
    gui.run_app()
