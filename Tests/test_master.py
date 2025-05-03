from inference_sdk import InferenceHTTPClient
import os

try:
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com/",  # use local inference server
        api_key='3QIvu7zIFiIAijMbDe5X'  # optional to access your private data and models
    )
    print(f"Connected to Roboflow API successfully.")
except Exception as e:
    print(f"Error connecting to Roboflow: {e}")
    raise

def run_master(input_path, source_type):
    """
    Runs the master.py script with the given input.

    Arguments:
    input_path -- Path to the input image or video file.
    source_type -- The type of source: 'V' for video, 'I' for image.
    fps -- Frames per second for video processing (optional).
    """
    # Absolute path to master.py
    master_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../master.py"))

    # Build the command
    command = f"python {master_script_path} -p {input_path} {source_type}"

    # Run the command
    print(f"Running: {command}")
    result = os.system(command)
    if result == 0:
        print(f"master.py ran successfully for {input_path}.")
    else:
        print(f"Error running master.py for {input_path}.")

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        {"input_path": "lightsaber.jpg", "source_type": "I"},
        {"input_path": "sample_video.mp4", "source_type": "V", "fps": 2},
    ]

    # Run tests
    for test in test_cases:
        run_master(**test)



