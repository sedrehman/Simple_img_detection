"""
Image Filtering
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with image filtering and familiarize you with 'tricks', e.g., padding, commonly used by computer vision 'researchers'.

Please complete all the functions that are labelled with '# TODO'. Steps to complete those functions are provided to make your lives easier. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
are building blocks you could use when implementing the functions labelled with 'TODO'.

I strongly suggest you read the function 'zero_pad' and 'crop' that are defined in 'utils.py'. You will need them!

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""
#syed rehman
import argparse
import copy
import os

import cv2 
import numpy as np

import utils

# low_pass filter and high-pass filter
low_pass = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]
high_pass = [[-1/8, -1/8, -1/8], [-1/8, 2, -1/8], [-1/8, -1/8, -1/8]]
#high_pass = [[-1/9 , -1/9 , -1/9], [-1/9 ,17/9, -1/9], [-1/9, -1/9, -1/9]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task1.jpg",
        help="path to the image"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="high-pass",
        choices=["low-pass", "high-pass"],
        help="type of filter"
    )
    parser.add_argument(
        "--result-saving-dir",
        dest="rs_dir",
        type=str,
        default="./results/",
        help="directory to which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)

def convolve2d(img, kernel):


    # my stuff 
    """Convolves a given image and a given kernel.

    Steps:
        (1) flips the kernel
        (2) pads the image # IMPORTANT
            this step handles pixels along the border of the image, and ensures that the output image is of the same size as the input image
        (3) calucates the convolved image using nested for loop

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), convolved image.
    """
    # TODO: implement this function.

    width = len(img)
    kernelFlipped = utils.flip_x(kernel)
    kernelFlipped = utils.flip_y(kernel)
    paddedImg = utils.zero_pad(img, 1, 1)
    newImg = [[0 for x in range(width)] for y in range(width)]
    
    for i in range(1, width):  #x
        for j in range(1, width):  #y

            w= 3
            h = 3
            temp = [[0 for x in range(w)] for y in range(h)] 
            # temp = []
            start_x = 0
            for k in range(i-1, i+2):
                
                start_y = 0
                for l in range(j-1, j+2):
                    temp[start_x][start_y] = paddedImg[k][l]
                    start_y += 1
                start_x += 1
                    
            tempConvoluted= utils.elementwise_mul(temp, kernelFlipped)
            total = 0
            for k in range(0, 3):
                for l in range(0, 3):
                    total += tempConvoluted[k][l]
            newImg[i][j] = total
    



    #elementwise_add to add everthing up in matrix.
    

    return newImg


    raise NotImplementedError

def main():
    args = parse_args()

    img = read_image(args.img_path)

    if args.filter == "low-pass":
        kernel = low_pass
    elif args.filter == "high-pass":
        kernel = high_pass
    else:
        raise ValueError("Filter type not recognized.")

    if not os.path.exists(args.rs_dir):
        os.makedirs(args.rs_dir)

    filtered_img = convolve2d(img, kernel)
    write_image(filtered_img, os.path.join(args.rs_dir, "{}.jpg".format(args.filter)))


if __name__ == "__main__":
    main()
