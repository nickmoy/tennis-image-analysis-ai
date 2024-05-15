# Tennis Image Analyzer

![output_video](https://github.com/nickmoy/tennis-image-analysis-ai/assets/25787918/8130e351-47df-4a7a-aacb-7b9cb10cc947)

Python project which tracks the positions of the players and the ball of an
image/video of a Tennis match. Uses YOLO models to track the players and the ball
and uses the Pytorch Resnet50 neural network to track keypoints on the court which are
used as reference points to calculate the positions of the players and the ball on the court
as well as their speeds.

I also draw a mini version of the court which displays the player and ball positions, although
the position for the ball is currently broken and the speeds are off, but I will fix that in the future.

All credit for the idea and structuring of the project goes to [abdullahtarek](https://www.github.com/abdullahtarek)
as I followed his Youtube
tutorial at [https://www.youtube.com/watch?v=L23oIHZE14w](https://www.youtube.com/watch?v=L23oIHZE14w)
and used his code at [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) as a reference.
Much of my code is very similar or identical to his except for some restructuring.

## How to run
To run the project you can download my pre-trained models and place them in the model folder or use the jupyter
notebooks provided in the training folder to train them yourself. The project also uses the players heights to convert
the on-screen coordinates to real positions on the court so you need to input the heights in meters of the two
players into the __init__.py file in the constants folder with Player 1 being the one closer to the camera.
Then you need to edit the main.py file's variable called "input_video_path" to the path of your image/video.

Then you can simply run `python main.py` and the output will be written to the output folder (You may need to create a
folder specifically called "output" first).

## Procedure
The project requires three models: one for the players, the ball, and the keypoints. The model I used
for the players is simply the out-of-the-box YOLOv8 model. The ball is much harder to detect and requires more
training, so there is a jupyter notebook in the training folder which must be run to generate a better YOLO model
on the dataset [here](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection).
Likewise, the keypoints are detected by training the Resnet50 Pytorch model on this
[dataset](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file) in a separate jupyter notebook.
All of the datasets are already downloaded and processed automatically through the jupyter notebooks but
can be downloaded directly below as well.

## Trained Models
* Player detector model: [YOLOv8](https://drive.google.com/file/d/1Im8gxQa4aD4PJuYLVZ8E8KO0QqrXdAQ1/view?usp=sharing)
* Ball detector model: [Trained YOLOv5](https://drive.google.com/file/d/1uScUqgg3Gyr0h7TC464mNpO_1kfR5SCI/view?usp=sharing)
* Keypoint detector model: [Trained Resnet50](https://drive.google.com/file/d/11BVpgXibfeh7zC0-JOG1T3a08VmIL2xt/view?usp=sharing)

## Datasets
* Ball detection dataset: [https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection) 
* Keypoint detection dataset: [https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file)

## Libraries Used
* Ultralytics
* YOLO
* Pytorch
* OpenCV
* Pandas
* Numpy
* Pickle
