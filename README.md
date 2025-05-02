# ECEN-4273 Project 2

## Overview
This project is part of the ECEN-4273 course and focuses on implementing and demonstrating key concepts in software engineering and project management. 
The codebase contains the implementation of an object detection pipeline using Roboflow's Inference SDK. 
The project processes images, video files, or live webcam feeds to detect objects and display results with bounding boxes and confidence levels on predictions.
Our model classifies 6 different objects:
- People
- Cats
- Dogs
- Sith Lightsabers
- Jedi Lightsabers
- Daleks
It achieves a mAP50 of ~81%. Indicating accuracy is fairly high despite our limited dataset size. 

## Features
- Object detection on images, videos, and webcam feeds.
- Multiple objects detected in a single frame, including real-time
- Real-time display of inference results using OpenCV.
- Post-processing to merge video frames into an `.mp4` file.
- Configurable frame rate for video processing.

## File Structure
- `master.py`: Main script for running the object detection pipeline.
- `requirements.txt`: List of dependencies for the project.
- `.github/workflows/CI_CD.yml`: GitHub Actions workflow for CI/CD.
- `LICENSE`: MIT License for the project.

## Prerequisites
- Python version < 3.12, >= 3.8 (Our demo ran on python 3.11.9)
- Required Python libraries (listed in `requirements.txt`).
- OpenCV installed (`pip install opencv-python`).
- A valid Roboflow API key for accessing the inference SDK.

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/JHoodCS/ECEN-4273-Project2.git
    ```
2. Navigate to the project directory:
    ```
    cd ECEN-4273-Project2
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Run the script with the following commands:

### Process a Video
    ```
    python master.py -p path/to/video.mp4 -f 2 V
    ```
    - `-p`: Path to the video file.
    - `-f`: Frame rate for processing.
    - `V`: Indicates video input.

### Process an Image
    ```
    python master.py -p path/to/image.jpg I
    ```
    - `-p`: Path to the image file.
    - `I`: Indicates image input.

### Process Webcam Feed
    ```
    python master.py W
    ```
    - `W`: Indicates webcam input.

2. For video input, the processed frames are merged into a single `.mp4` file named `output.mp4`.
3. For image input, the processed image is displayed in a window.
4. For webcam input, the processed frames are displayed in real-time.

## Autodocumentation
We used pydoc to generate our documentation. To regenerate the documentation use:
    ```
    python -m pydoc master > helpfile.txt
    ```
## Notes
- Ensure that the API key in `master.py` is valid for accessing Roboflow's services.
- Modify the `requirements.txt` file if additional dependencies are needed.
- The `output.mp4` file will be saved in the current working directory.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
