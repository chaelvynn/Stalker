# import Tkinter to create our GUI.
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
# import openCV for receiving the video frames
import cv2
# make imports from the Pillow library for displaying the video stream with Tkinter.
from PIL import Image, ImageTk
# Import the tello module
from djitellopy import tello
# Import threading for our takeoff/land method
import threading
# import our flight commands
from flight_commands import start_flying, stop_flying


# Class for controlling the drone via keyboard commands
class DroneController:
    def __init__(self):
        # Initialize the Tkinter window, give it a title, and define its minimum size on the screen.
        self.root = Tk()
        self.root.title("Drone Keyboard Controller - Tkinter")
        self.root.minsize(800, 600)

        # Create a hidden frame to handle input from key presses and releases
        self.input_frame = Frame(self.root)

        # Initialize, connect, and turn on the drones video stream
        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()
        # Initialize a variable to get the video frames from the drone
        self.frame = self.drone.get_frame_read()

        # Define a speed for the drone to fly at
        self.drone.speed = 50

        # Label for displaying video stream
        self.cap_lbl = Label(self.root)

        # Frame to hold buttons
        self.button_frame = Frame(self.root)

        # Create a button to send takeoff and land commands to the drone
        self.takeoff_land_button = Button(self.button_frame, text="Takeoff/Land", command=self.takeoff_land)

        # Create a button for dropdown
        self.dropdown_button = Button(self.button_frame, text="Dropdown", command=self.show_dropdown)

        # Variable for dropdown menu
        self.dropdown_var = StringVar(self.root)
        self.dropdown_var.set("Option 1")

        # Create a dropdown menu
        self.dropdown_menu = OptionMenu(self.button_frame, self.dropdown_var, "Option 1", "Option 2", "Option 3")

    # Define a method for taking off and landing
    def takeoff_land(self):
        # Set the command for taking off or landing by checking the drones is_flying attribute
        if self.drone.is_flying:
            threading.Thread(target=self.drone.land).start()
        else:
            threading.Thread(target=self.drone.takeoff).start()

    # Dummy method to show dropdown (no functionality for now)
    def show_dropdown(self):
        pass

    # Method to run the application
    def run_app(self):
        try:
            # Pack the hidden frame and give direct input focus to it.
            self.input_frame.pack()
            self.input_frame.focus_set()

            # Pack the video label
            self.cap_lbl.pack(anchor="center", pady=15)

            # Add the buttons to the button frame and pack it below the video
            self.takeoff_land_button.pack(side='left', padx=10)
            self.dropdown_button.pack(side='left', padx=10)
            self.dropdown_menu.pack(side='left', padx=10)
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

        # Read a frame from our drone
        frame = self.frame.frame

        frame = cv2.resize(frame, (w, h))

        # Convert the current frame to the rgb colorspace
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

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
            self.drone.end()
            self.root.quit()  # Quit the Tkinter main loop
            exit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    # Initialize the GUI
    gui = DroneController()
    # Call the run_app method to run tkinter mainloop
    gui.run_app()
