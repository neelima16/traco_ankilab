# TRACO repository
Hello to the TRACO Seminar Repository,
here you will find everything that you need to train and test your model.

## Data
We give you a reference solution to show you one very crapy way how you could do it. 
There is a requirements.txt which lets you create a suitable env.
In the folder training is all the training data for you model.
In the folder leaderboard data are 5 videos. For each you should creat a prediction csv and upload them to our website to get a leaderboard score.
There are also the submission guidelines to help you upload the csv.

## Annotation GUI
The annotation GUI is available in the python file `traco_annotation_gui.py`. It can be run with the following command:

```python traco_annotation_gui.py```

We recommend using Python 3.6 or higher. The GUI requires the following packages:
- PyQt6
- Pandas
- imageio (```pip install imageio-ffmpeg```)
- pyqtgraph
- matplotlib
- numpy

Annotate the videos by left-clicking on the head of a HexBug. Note that you have to select the HexBug ID on the right 
side of the GUI before annotating. You can also use A and D keys to move the video forward and backward frame by frame.

## Score
The file get_score.py lets you evaluate your model to see which score you would get. For your better understanding we added a log and if a video is available you can also crate a video for better understanding:
      - if you want to have a log for your score calculation set log = True
      - if you want to have a video of you score calculation set vid = True
        -> csv must be same number of frames and there has to be a mp4 video in the same folder with the groud trouth
You can call it like this: score = get_score(str(pred_path / file), str(gt_path / file), log=True, cid = False)

## Reference Solution
We suggest you to run the solution on Google Colab and connect that to you google drive in order to load the model and the videos.