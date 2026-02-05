import cv2
import numpy as np

def detect_shapes_and_centroids(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth out the texture noise
    # This prevents the texture from being detected as edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use Canny Edge Detection
    # Thresholds are chosen to catch shape boundaries but ignore texture
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate the edges to ensure contours are closed
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        # Calculate Area
        area = cv2.contourArea(cnt)

        # Filter:
        # 1. Remove small noise (area < 500)
        # 2. Remove the large background "hill" boundary (area > 10000)
        if area < 500 or area > 10000:
            continue

        # Calculate Centroid using Moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Shape Identification
        # 1. Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        shape_name = "Unknown"

        # 2. Classification Logic
        if vertices == 3:
            shape_name = "Triangle"

        elif vertices == 4:
            # Aspect Ratio check for Square
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # A square has an aspect ratio close to 1
            shape_name = "Square" if 0.9 <= ar <= 1.1 else "Rectangle"

        else:
            # Distinguish Star vs Circle using Solidity
            # Solidity = Contour Area / Convex Hull Area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = area / float(hull_area)
                
                # Stars are concave (have 'arms'), so they have low solidity (< 0.8)
                # Circles are convex, so they have high solidity (> 0.9)
                if solidity < 0.85:
                    shape_name = "Star"
                else:
                    shape_name = "Circle"

        # Append to results list as requested
        results.append([shape_name, (cx, cy)])

        # Visualization (Optional: Draw name and centroid on image)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, shape_name, (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Print the final list
    print("Detected List [Shape, (Centroid X, Centroid Y)]:")
    for item in results:
        print(item)

    # Display the result
    cv2.imshow("Identified Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function with your image file name
detect_shapes_and_centroids('photo_2.jpg')