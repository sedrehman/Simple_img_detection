"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment 
the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'and the 
functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""
#syed rehman
#import cv2
import argparse
import json
import os

import utils
#from task1 import *
import task1 as t1

template_avg = 0.0
template_after_subtraction = []

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def getMean(temp):
    height = len(temp)
    width = len(temp[0])
    total = 0
    numOfDigits = 0
    for y in range(height):
        for x in range(width):
            total += temp[y][x]
            numOfDigits += 1
    avg = float(total/numOfDigits)
    return avg


def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:

    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) 
    ---------------------------------------------------------------------------------
    (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5

    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    global template_after_subtraction
    height = int(len(patch))
    width = int(len(patch[0]))
    patch_avg = float(getMean(patch))

    # print("patch height : ", height)
    # print("pathc width : ", width)
    # print("template height : ", len(template))
    # print("template width : ", len(template))

    ncc = 0.0
    numerator = 0.0
    denominator = 0.0

    """
    numerator = sum( ( template[y][x] - tamplate_avg ) * ( patch[y][x] - patch_avg ))

    denominator = sqrt(sum((template[y][x] - tamplate_avg)^2) * sum((patch[y][x] - patch_avg)^2))

    ncc = numerator/ denominator
    """

    for y in range(height):
        for x in range(width):
            a = float(patch[y][x] - patch_avg)
            #b = float(template[y][x] - template_avg)
            #b = float(template_after_subtraction[y][x])
            numerator += float(a * (template_after_subtraction[y][x]))
    
    a_sum = 0.0
    b_sum = 0.0
    for y in range(height):
        for x in range(width):
            a = float( float(patch[y][x] - patch_avg)**2 )
            a_sum += a
            #b = float(float(template[y][x] - template_avg)**2)
            b = float(template_after_subtraction[y][x]**2)
            #b = float(template[y][x] - template_avg)
            b_sum += b

    denominator = float((a_sum * b_sum)**.5)
    ncc = float(numerator / denominator)

    return ncc
    #raise NotImplementedError


"""
def match(img, template):
    Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    



    # TODO: implement this function.
    # raise NotImplementedError
    template_avg = float(getMean(template))
    img_height = len(img)
    img_width = len(img[0])
    
    template_width = int(len(template[0]))
    template_height = int(len(template))

    padded_top = 0
    padded_botton = 0
    padded_right = 0
    padded_left = 0
    padded_height = template_height + img_height - 1
    padded_width = template_width + img_width - 1

    if (template_height % 2) == 0:
        padded_top = template_height / 2  #top is half 
        padded_botton = (template_height / 2) - 1   #bottom is one less than half
    else:
        padded_top = round(template_height / 2)  #always rounds up for .5 = 1
        padded_botton = int(template_height - padded_top)
    
    if (template_width % 2) == 0:
        padded_left = (template_width / 2)
        padded_right = (template_width / 2) - 1
    else:
        padded_left = round(template_width / 2)  #always rounds up for .5 = 1
        padded_right = template_width - padded_top

    padded_img = [[0 for x in range(padded_width)] for y in range(padded_height)]
    i = 0
    for y in range(int(padded_top), int(padded_top + img_height)):
        j = 0
        for x in range(int(padded_left), int((padded_left + img_width))):
            padded_img[y][x] = img[i][j]
            j += 1
        i += 1

    #padding done.
    highest_ncc = 0.0
    x_pos = int(0)
    y_pos = int(0)
    i = 0
    j = 0
    for y in range(int(padded_top), int(padded_top + img_height)):
        for x in range(int(padded_left), int(padded_left + img_width)):
            #going through each coordinate
            patch = []
            for i in range( int(y - padded_top), int(y + padded_botton + 1)):
                temp = []
                for j in range( int(x - padded_left) , int(x + padded_right + 1)):
                    temp.append(padded_img[i][j])
                patch.append(temp)
            value = float(norm_xcorr2d(patch, template))
            if (value > highest_ncc):
                highest_ncc = value
                x_pos = int(x - padded_left)
                y_pos = int(y - padded_top)

    return (x_pos, y_pos, highest_ncc)
    raise NotImplementedError
"""
def match(img, template):
    global template_after_subtraction
    global template_avg
    template_avg = float(getMean(template))
    img_height = len(img)
    img_width = len(img[0])
    
    template_width = int(len(template[0]))
    template_height = int(len(template))
    for y in range(template_height):
        temp = []
        for x in range(template_width):
            temp.append(float(template[y][x] - template_avg))
        template_after_subtraction.append(temp)

    x_pos = 0
    y_pos = 0
    highest_ncc = 0.0
    i = 0
    j = 0
    for y in range(img_height-template_height):
        for x in range(img_width - template_width):
            patch = []
            for i in range(y, y+ template_height):
                temp = []
                for j in range(x, x+template_width):
                    temp.append(img[i][j])
                patch.append(temp)
            value = float(norm_xcorr2d(patch, template))
            if (value > highest_ncc):
                highest_ncc = value
                x_pos = int(x)
                y_pos = int(y)

    return (y_pos, x_pos, highest_ncc)
    #raise NotImplementedError

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = t1.read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = t1.read_image(args.template_path)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    #print("x : " ,x , " y :", y , "max ncc: ", max_value )
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
