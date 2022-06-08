# Ball Tracking Using Curve Fitting

## Description

Given a video of a moving ball, this project finds the trajectory of the ball by tracking it and using the Standard Least Squares Method.


## Data

Two videos of a red moving ball are considered here. The first one is noise free, while the second one has some noise with respect to the movement of the ball.

![alt text](/output/ball1.gif)

![alt text](/output/ball2.gif)


## Approach

* Each frame from the video is extracted and converted from RBG to HSV color space.

* Each frame is masked for the red color.

* The coordinates of these contours are extracted and the top and bottom points are stored considering the max and min values.

* Considering all of these points, Curve Fitting is done using the Normal Equation of the Standard Least Squares Method.


## Output

![alt text](/output/ball1out.png)

![alt text](/output/ball2out.png)


## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>
<a href="https://matplotlib.org/" target="_blank" rel="noreferrer"> <img src="https://static.javatpoint.com/tutorial/matplotlib/images/matplotlib-tutorial.png" alt="matplotlib" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/Ball-Tracking-Using-Curve-Fitting.git
```

* Open the repository and navigate to the `src` folder.
```
cd Ball-Tracking-Using-Curve-Fitting/src
```

* Run the program.
```
python main.py
```


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)
