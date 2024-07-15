# Stalker
_Stalker_ is a project that aims to develop a Ground Control System (GCS) using DJI Tello drone, enabling it to follow a person automatically. The system allows the pilot to control the drone either manually or automatically.

## Features
- Manual Control
  - Control the drone by using a keyboard.
- Automatic
  - Using face recognition to identify and follow a specific person
  - Using optical flow to track the person afterward
- GCS
  - A live video stream from the drone
  - Drone controller
  - Choosing a target to be followed

## Requirements
- DJI Tello Drone
- Python 3.x
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/chaelvynn/Stalker.git
   cd Stalker/Interface
   ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## How to use
1. Connect your current device to the Tello drone's Wi-fi.
2. Run `DroneController.py`.
