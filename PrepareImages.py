import cv2
import os
import numpy as np

image_folder_path = "C:\\Users\\gross\\Documents\\Data\\frogs"
image_folder_union = image_folder_path + "_union"
sum_width_px = 0
sum_height_px = 0
image_amount = 0
for file_path in os.listdir(image_folder_path):
    try:
        img = cv2.imread(image_folder_path + "\\" + file_path,0)
        #cv2.imshow('image',img)
        #cv2.waitKey(1)
        print("Read Image " +  str(image_amount))
        height, width = img.shape
        sum_width_px = sum_width_px + width
        sum_height_px = sum_height_px + height
        image_amount = image_amount + 1
    except:
        pass

avg_width_px = int(sum_width_px / image_amount)
avg_height_px = int(sum_height_px / image_amount)
dim = (avg_height_px, avg_width_px)
print("Width: " + str(avg_width_px) + "\n" + "Height:" + str(avg_height_px))
try:
    os.makedirs(image_folder_union)
except:
    pass

image_amount = 0
for file_path in os.listdir(image_folder_path):
    try:
        img = cv2.imread(image_folder_path + "\\" + file_path,0)
        #cv2.imshow('image',img)
        #cv2.waitKey(1)
        print("Resized Image " +  str(image_amount))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #cv2.imshow("resized", resized)
        #cv2.waitKey(1)
        cv2.imwrite(image_folder_union + "\\"+file_path,resized)
        image_amount = image_amount + 1
    except:
        pass


