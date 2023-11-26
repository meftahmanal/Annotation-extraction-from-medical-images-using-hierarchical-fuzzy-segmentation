import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import skimage.morphology


def process_image(image_path, inverse=False):
    # Gaussian filter
    image1 = cv2.imread(image_path)

    if inverse:
        image1 = 255 - image1
    #grayscale
    gray = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)


    #removing small noises with morphological opening

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    #clahe contrast enhancement

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    cl1 = clahe.apply(opening)

    normalized_image = (cl1 - np.min(cl1)) / (np.max(cl1) - np.min(cl1)) * 255

    def fuzzy_entropy_threshold(image):
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
        probs = hist / np.sum(hist)

        H_max = -np.inf
        a_opt = 0
        c_opt = 0

        for a in range(0, 255):
            for c in range(a + 1, 256):
                k_dark = np.clip((c - np.arange(256)) / (c - a), 0, 1)
                k_bright = np.clip((np.arange(256) - a) / (c - a), 0, 1)

                P_dark = np.sum(k_dark * probs)
                P_bright = np.sum(k_bright * probs)

                H = -P_dark * np.log2(P_dark + 1e-6) - P_bright * np.log2(P_bright + 1e-6)

                if H > H_max:
                    H_max = H
                    a_opt = a
                    c_opt = c

        threshold = (a_opt + c_opt) // 2

        binary_image = (image >= threshold).astype(np.uint8) * 255

        return binary_image
    # calculate the optimal entropy for the image
    binary_image = fuzzy_entropy_threshold(normalized_image)
    

    labels = label(binary_image, connectivity=2)  # 2 in 2D image, equivalent to 8-connectivity

    regions = regionprops(labels)

    properties = ['area','convex_area','bbox_area', 'extent',  
                'mean_intensity', 'solidity','eccentricity']
    pd.DataFrame(regionprops_table(labels, image1, 
                properties=properties))
    masks = []
    bbox = []
    list_of_index = []

    min_area = 100
    max_solidity = 1
    min_eccentricity = 0.5
    max_area= 1000
    
    #define caracteristics for regions to get rid off

    for num, x in enumerate(regions):
        area = x.area
        solidity = x.solidity
        eccentricity = x.eccentricity
        if (num != 0 and area > min_area and area< max_area and  solidity> 0.6 and solidity < max_solidity and eccentricity > min_eccentricity):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
            

    count = len(masks)


    mask = np.zeros_like(labels)
    for i, x in enumerate(list_of_index):
        mask = (labels == x + 1).astype(np.uint8)

    #filter stuff we don't want
    masked_image = gray *(1-mask)


    labels1 = label(masked_image, connectivity=2)  # 2 in 2D image, equivalent to 8-connectivity'''
    
    def auto_canny_edge_detection(image, sigma=0.25,area_threshold=150):
        import skimage
        md = np.median(image)
        lower_value = int(max(0, (1.0-sigma) * md))
        upper_value = int(min(255, (1.0+sigma) * md))
        edges = cv2.Canny(image, lower_value, upper_value)
        out = skimage.morphology.area_opening(edges, area_threshold, connectivity=2)
        return out
    masked_image1 = masked_image.astype(np.uint8)

    edges = auto_canny_edge_detection(masked_image1)

    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 150:  # Add your desired minimum contour area threshold
            # Calculate mean intensity within the contour region
            mask = np.zeros_like(edges)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_intensity = np.mean(masked_image1[mask > 0])

            # Calculate solidity of the contour
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            if mean_intensity > 30 and solidity > 0.4:
                n+=1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image1, f"Arrow {n}:", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(image1, f"({x}, {y})", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    return n,binary_image, image1

def display_images_side_by_side(images1, images2, titles):
    n = len(images1)
    fig, axes = plt.subplots(2, n, figsize=(n * 6, 8))  # Increase the size of the figure by adjusting the figsize parameter

    for i in range(n):
        axes[0, i].imshow(images1[i], cmap='gray')
        axes[0, i].set_title(titles[i])
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        axes[1, i].imshow(images2[i], cmap='gray')
        axes[1, i].set_title('Inverse ' + titles[i])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()  # Adjust the spacing between subplots automatically
    plt.show()




def main():
    # Define the path to the desired directory
    input_directory = 'selFiles-1/'
    #Define desired 
    allowed_extensions = ('.jpg','.jpeg','.pgm')

    for file in os.listdir(input_directory):
        if file.lower().endswith(allowed_extensions):
            input_image_path = os.path.join(input_directory, file)
            print(f"Processing {file}")

            n_original, binary_original,  image_original = process_image(input_image_path)
            n_inverse, binary_inverse,  image_inverse = process_image(input_image_path, inverse=True)

            titles = ['Binary Image', 'Arrows']

            # Convert binary images from grayscale to color
            binary_color_original = cv2.cvtColor(binary_original, cv2.COLOR_GRAY2BGR)
            binary_color_inverse = cv2.cvtColor(binary_inverse, cv2.COLOR_GRAY2BGR)

            # Set the minimum and maximum contour thresholds
            min_contours = 1
            max_contours = 50

            # Check the number of contours detected in each image
            if min_contours <= n_original <= max_contours and min_contours <= n_inverse <= max_contours:
                # Select the image with the highest number of contours
                if n_original >= n_inverse:
                    print("Original Image Selected")
                    print(plt.imshow(image_original))

                    plt.axis('off')

                    display_images_side_by_side([binary_color_original,   image_original],
                                                [binary_color_inverse,   image_inverse],
                                                titles)

                else:
                    print("Inverse Image Selected")
                    print(plt.imshow(image_inverse))
                    plt.axis('off')

                    display_images_side_by_side([binary_color_inverse, image_inverse],
                                                [binary_color_original, image_original],
                                                titles)
            elif min_contours <= n_original <= max_contours:
                print("Original Image Selected")
                print(plt.imshow(image_original))
                plt.axis('off') 
                display_images_side_by_side([binary_color_original, image_original],
                                            [binary_color_inverse,   image_inverse],
                                            titles)

            elif min_contours <= n_inverse <= max_contours:
                print("Inverse Image Selected")
                print(plt.imshow(image_inverse))
                plt.axis('off')
                display_images_side_by_side([binary_color_inverse,   image_inverse],
                                            [binary_color_original, image_original],
                                            titles)
            else:
                if n_original >= n_inverse:
                    print("Original Image Selected")
                    print(plt.imshow(image_original))
                    plt.axis('off')
                    display_images_side_by_side([binary_color_original,    image_original],
                                                [binary_color_inverse,   image_inverse],
                                                titles)
                else:
                    print("Inverse Image Selected")
                    print(plt.imshow(image_inverse))
                    plt.axis('off')
                    display_images_side_by_side([binary_color_inverse,   image_inverse],
                                                [binary_color_original,  image_original],
                                                titles)

if __name__ == "__main__":
    main()
