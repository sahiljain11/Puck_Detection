# Summary

In project, I used a camera stream in order to detect the location of a puck on an air hockey table.
To do this, I used OpenCV and a HSV gaussian mask to isolate the puck and then used a set of stretches
to reshape the shape of the field.

# Usage

To use this code, I used the argparse python library to simplify this process. I've listed the different arguments below:

```
-h, --help            show this help message and exit
-s STREAM, --stream STREAM
                      stream video from videostream. denote capture number
                      (default=0)
-w WRITE, --write WRITE
                      write video stream to file (default="output.avi")
-r READ, --read READ  read from existing video file
-d DEBUG, --debug DEBUG
                      debug by piping extra info into debug.txt
```

When trying to use a video stream, use the '-s' command. This value will default to the camera at location 0. If there is an error
in this, it's likely due to the configuration of the camera with the operating system itself. I had slight trouble getting the
playstation camera to work on Windows, but it worked just fine on Linux.

To use an already existing video stream, use the '-r' command and specify what video file you wish for the code to read. I believe
that in the code currently, I overwrote this functionality as I was doing some severe debugging, but to add the functionality back,
simply remove `vidcap = cv2.VideoCapture("playstationcam1_fixed.avi")` from the BlobDetection.py file.

To run the code, simply run the BlobDetection.py file in your terminal with the proper arguments. Below is an example:
```
python BlobDetection.py -s 0
```
# Important Note
Ensure that you have a proper visualization method via the terminal. Personally, I used a Windows Linux Subsystem and used XMING
and simply exported the visualization parameters at the prior to running the code. I believe PyCharm does this automatically but I'm not sure if you can run argparse in PyCharm.

Other than that, enjoy!
