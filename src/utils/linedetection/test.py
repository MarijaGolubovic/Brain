import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r'D:\Documents\BOSCH\test2.jpg'
image1 = cv2.imread(image_path)
plt.imshow(image1)

filename = 'picam2023_1_22_10_12_14.h264'
file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 0.6  # Option to scale to fraction of original size.

# We want to save the output to a video file
output_filename = 'res.mp4'
output_frames_per_second = 20.0


def grey(image):
    # convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges


def region(image):
    height, width = image.shape
    # isolate the gradients that correspond to the lane lines
    polygon = np.array([
        [(2, 680), (width, 680), (70, 300), (850, 300)]
    ])
    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillConvexPoly(mask, polygon, 255)
    mask = cv2.fillConvexPoly(mask, np.array([(2, 680), (70, 300), (474, 470)]), 255)
    mask = cv2.fillConvexPoly(mask, np.array([(width, 680), (850, 300), (474, 470)]), 255)
    #plt.imshow(mask)
    #plt.title('Triangle'), plt.xticks([]), plt.yticks([])
    #plt.show()
    isreg = cv2.bitwise_and(image, mask)
    return isreg


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def average(image, lines):
    left = []
    right = []

    if lines is None:
        return None
    for line in lines:
        #print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #print('x1: ', x1, 'x2: ', x2, 'y1: ', y1, 'Y2: ', y2)
        # fit line to points, return slope and y-int
        A1 = np.array([x1, x2])
        A2 = np.array([y1, y2])
        #idx = np.isfinite(A1) & np.isfinite(A2)
        #print('A1 ', A1,' A2 ', A2, ' idx ', idx)
        #print(y2-y1)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters)
        #print('a1: ', A1, ' a2: ', A2,'0: ',parameters[0], '1: ', parameters[1])
        #print('1: ', parameters[1])
        print('par: ', parameters)
        slope = parameters[0]
        y_int = parameters[1]
        # lines on the right have positive slope, and lines on the left have neg slope
        if slope <= 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    # takes average among all the columns (column0: slope, column1: y_int)
    print(left)
    print(right)
    try:
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        # create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])
    except:
        print("no line")




def make_points(image, average):
    #print(average)
    slope, y_int = average
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * 0.6)
    # determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


#from google.colab.patches import cv2_imshow

'''##### DETECTING lane lines in image ######'''

# Load a video
cap = cv2.VideoCapture(filename)

    # Create a VideoWriter object so we can save the video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter(output_filename,
                             fourcc,
                             20,
                            (1280, 720))

    # Process the video
while cap.isOpened():

    # Capture one frame at a time
    success, frame = cap.read()

        # Do we have a video frame? If true, proceed.
    if success:
        # Resize the frame
        width = int(frame.shape[1] * scale_ratio)
        height = int(frame.shape[0] * scale_ratio)
        frame = cv2.resize(frame, (width, height))
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
    #else:
        #break

        #original_frame = frame.copy()


    #copy = np.copy(image1)
        #copy = np.copy(frame)
        gray = grey(frame)
        blur = gauss(gray)
        edges = cv2.Canny(blur, 50, 150)
        #plt.imshow(edges)
        #plt.show()
        isolated = region(edges)
        #cv2.imshow('edges', edges)
        #cv2.imshow('isolated', isolated)
        #cv2.waitKey(1)
        """
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        plt.imshow(isolated, cmap='gray')
        plt.title('Isolated region'), plt.xticks([]), plt.yticks([])
        plt.show()
        """
        """""
        # DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
        """

        lines = cv2.HoughLinesP(isolated, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        

        try:
            averaged_lines = average(frame, lines)
            black_lines = display_lines(frame, averaged_lines)
            cv2.imshow('lines', black_lines);
            cv2.waitKey(1)
            # taking wighted sum of original image and lane lines image
            lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
            result.write(lanes)
        except:
            result.write(frame)

        """plt.imshow(lanes, cmap='gray')
        plt.title('Lanes'), plt.xticks([]), plt.yticks([])
        plt.show()"""
        #cv2.waitKey(0)

        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break

    # No more video frames left
    else:
        break

    # Stop when the video is finished
cap.release()

# Release the video recording
result.release()

# Close all windows
cv2.destroyAllWindows()
#cv2.imshow(lanes)
#cv2.waitKey(0)