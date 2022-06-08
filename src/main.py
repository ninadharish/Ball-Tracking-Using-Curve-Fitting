import cv2
import numpy as np
import matplotlib.pyplot as plt


def trajectory(video):
    """Function to find the trajectory of the ball

    Args:
        video: Video of the moving ball to be tracked

    Returns:
        Trajectory Information
    """

    cap = cv2.VideoCapture(video)
    top_list = []
    bottom_list = []

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([155,25,0])
        upper = np.array([179,255,255])
        mask = cv2.inRange(frame, lower, upper)

        image_height, _ = mask.shape

        cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 2:
            cnts = cnts[0]
        else:
            cnts = cnts[1]

        c = max(cnts, key=cv2.contourArea)

        top = list(c[c[:, :, 1].argmin()][0])
        bottom = list(c[c[:, :, 1].argmax()][0])

        top_list.append(top)
        bottom_list.append(bottom)     

    cap.release()
    
    plot_start = bottom_list[0][0]
    plot_end = top_list[-1][0]

    points = np.array(top_list + bottom_list)

    X_curve = np.zeros((len(points),3))
    Y_curve = np.zeros(len(points))
    for i in range(len(points)):
        X_curve[i][0] = 1
        X_curve[i][1] = points[i][0]
        X_curve[i][2] = (points[i][0])*(points[i][0])
        Y_curve[i] = image_height - points[i][1]

    theta = np.dot(np.linalg.inv(np.dot(X_curve.T, X_curve)), np.dot(X_curve.T, Y_curve))
    cv2.destroyAllWindows()

    return theta, points, plot_start, plot_end, image_height


def plot(i, video):
    """Function to plot the trajectory

    Args:
        i: Plot number
        video: Video of the moving ball to be tracked
    """
    theta, points, plot_start, plot_end, image_height = trajectory(video)
    plt.figure(i)
    x_plot = np.linspace(plot_start, plot_end, 50)
    y_plot = theta[0]*1 + theta[1]*x_plot + theta[2]*x_plot*x_plot
    plt.scatter(points[:,0], (image_height - (points[:,1])))
    plt.plot(x_plot, y_plot, 'r')
    plt.title(video, fontsize=18)


if __name__ == "__main__":

    plot(1, './data/ball_video1.mp4') #Plot the trajectory of the ball for the first video in Figure 1
    plot(2, './data/ball_video2.mp4') #Plot the trajectory of the ball for the second video in Figure 2
    plt.show()