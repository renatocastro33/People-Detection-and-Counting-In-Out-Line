# People-Detection-and-Counting-In-Out-Line
A people detection using Yolo v3 and Deep Sort to count people who passes a certain line that delimits the entrance of a place. This program can count the poeple who get into the place or get out.

# Project based On
This project is based on the Deep-Sort Algorithm usimg Yolo v3: https://github.com/Qidian213/deep_sort_yolov3
What we are looking forward to resolve in this github is the counting from a line.

# Requirements
For CPU:

  <li>tensorflow>=1.4.0 (not compatible with tf 2.0) #This will be resolved soon</li>
  <li>NumPy</li>
  <li>sklean</li>
  <li>OpenCV</li>
  <li>Pillow</li>


For GPU:

  <li>tensorflow>=1.4.0 (not compatible with tf 2.0) #This will be resolved soon</li>
  <li>NumPy</li>
  <li>sklean</li>
  <li>OpenCV</li>
  <li>Pillow</li>
# Download pre-trained weights

Before testing the code you must download the weights for the deep learning model for the trackers and the yoloV3.

Drive Link for Deep Sort Weights : https://drive.google.com/open?id=1rjZpYVorVCDyx25BkkQ0rdr2wdnYM71X
Drive Link for Yolov3 weights: https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view
 <li>You must put the both files into the model_data folder</li>
 
# Test

Command Linux Terminal or Anaconda Prompt: 

  <li>python main.py -i path_input_video -n 1</li>

Test Video Results: https://www.youtube.com/watch?v=Cc-dRiBepCU&feature=youtu.be
