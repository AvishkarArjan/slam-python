# SLAM using python
based on [this](https://www.youtube.com/watch?v=7Hlb8YX2-W8) livestream.

- Feature based SLAM
- ORB opencv3

## Development Phases
- Feature Detection
![Feature Detection](/resources/feature_detection.png)

## Things to explore
Some of the questions I had while going through the stream - 
- Feature based SLAM vs Dense SLAM
- What are Keypoints and Descriptors ? ([Ans](https://answers.opencv.org/question/37985/meaning-of-keypoints-and-descriptors/))
- How to track good features in the frame ?

## Current Problems
- Feature detection uses a lot of CPU (~96% CPU usage on my Intel i5 11th Gen)
