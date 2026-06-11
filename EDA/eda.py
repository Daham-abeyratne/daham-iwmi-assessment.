#----Visual analysis---
# dataset has 2 folders as with_mask & without_mask
# most of the images are mostly zoomed in
# have pictures from different side


# withmask size-> 3725

# withoutmask size-> 3828

import cv2
import os
import numpy as np
from collections import Counter

dataset_path = "E:\\IWMI_Assesment\\data"

widths = []
heights = []
curropted = 0

#to loop through both folder and every image
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)

        img = cv2.imread(image_path)

        if img is None:
            curropted += 1
            continue 

        height, width = img.shape[:2]

        widths.append(width)
        heights.append(height)

print("Width Statistics")
print("Min Width:", min(widths))
print("Max Width:", max(widths))
print("Average Width:", np.mean(widths))
print("Median Width:", np.median(widths))

# Print height statistics
print("\nHeight Statistics")
print("Min Height:", min(heights))
print("Max Height:", max(heights))
print("Average Height:", np.mean(heights))
print("Median Height:", np.median(heights))

print("\nTotal Images Processed:", len(widths))
print("curropted:",curropted)

print("top 10 frequent demensions")
print("widths")
for width, count in Counter(widths).most_common(10):
    print(f"{width}: {count}")

print("Heights")
for height, count in Counter(heights).most_common(10):
    print(f"{height}: {count}")
