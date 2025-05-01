# This will be where functions are called only, there should be little to no clutter in this file
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient
import os
import argparse


rate = 2 #framerate for video

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path')
parser.add_argument('-f', '--fps')
parser.add_argument('source')

args = parser.parse_args()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # use local inference server
    api_key='3QIvu7zIFiIAijMbDe5X'  # optional to access your private data and models
)


def my_sink(result, video_frame):
    if result.get("output_image"):  # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)


project_id = "sep2-e2mjc"
model_version = 5
image_url = "lightsaber.jpg"
video_url = "row1.mp4"
output_video_name = "output.mp4"
result_list = []  # of type ndarray

# Merges images from result_list into an mp4 video

def image_merger():
    # Get the height and width of the video frames
    frame_height, frame_width, _ = result_list[0].shape
    # Set the encoding format of the video writer
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    # Create the VideoWriter
    out = cv2.VideoWriter(output_video_name, fourcc,
                          30.0, (frame_width, frame_height))
    # Write each frame in the result list to the output video
    for frame in result_list:
        out.write(frame)
    # Release the output (save the video)
    out.release()


def predict_and_display(source_url, source_type):
    if (source_type == "V"):
        def my_sink(result, video_frame):
            if result.get("output_image"):  # Display an image from the workflow response
                cv2.imshow("Workflow Image",
                           result["output_image"].numpy_image)
                # Add each frame of the video to result_list for later recombination (post prediction)
                result_list.append(result["output_image"].numpy_image)
                cv2.waitKey(1)
        # Run inference on video
        # initialize a pipeline object
        pipeline = InferencePipeline.init_with_workflow(
            api_key='3QIvu7zIFiIAijMbDe5X',
            workspace_name="sep2",
            workflow_id="identify-sep2",
            # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
            video_reference=source_url,
            max_fps=rate,
            on_prediction=my_sink
        )
        pipeline.start()  # start the pipeline
        pipeline.join()  # wait for the pipeline thread to finish
    elif (source_type == "I"):
        def plot_image(result):
            if "output_image" in result[0] and result[0]["output_image"]:
                output_image = result[0]["output_image"]
                img_bytes = base64.b64decode(output_image)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                cv2.imshow("Workflow Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        result = CLIENT.run_workflow(
            workspace_name="sep2",
            workflow_id="identify-sep2",
            images={
                "image": source_url
            },
            use_cache=True  # cache workflow definition for 15 minutes
        )
        # Call the sink function with the result and None for video_frame
        plot_image(result)
    elif (source_type == 'W'):
        def my_sink(result, video_frame):
            if result.get("output_image"):  # Display an image from the workflow response
                cv2.imshow("Workflow Image",
                           result["output_image"].numpy_image)
                cv2.waitKey(1)
        # Run inference on video
        # initialize a pipeline object
        pipeline = InferencePipeline.init_with_workflow(
            api_key='3QIvu7zIFiIAijMbDe5X',
            workspace_name="sep2",
            workflow_id="identify-sep2",
            # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
            video_reference=0,
            max_fps=30,
            on_prediction=my_sink
        )
        pipeline.start()  # start the pipeline
        pipeline.join()  # wait for the pipeline thread to finish
    else:
        print("Invalid source type. Please use 'V' for video, 'I' for image, or 'W' for webcam.")
if(args.source == 'W'):
    predict_and_display("NULL", 'W')
elif(args.source == 'V'):
    predict_and_display(args.path, args.source)
    rate = float(args.fps)
elif(args.source == 'I'):
    predict_and_display(args.path, args.source)

image_merger()  # Merge the predicted images back into a single .mp4 file
