import cv2
import numpy as np

def create_black_mask(image_path, output_path='mask_output.png', display_result=False):
    """
    Creates a black mask of the foreground object in an image using the GrabCut algorithm.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the output mask image.
    - display_result (bool): Whether to display the resulting mask.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image at path '{image_path}'.")
        return False

    # Initialize resize flag
    image_resized = False

    # Optionally, resize the image if it's too large
    max_dimension = 1024  # Maximum dimension to resize to
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        image_resized = True
        print(f"Image resized to {image.shape[1]}x{image.shape[0]} for faster processing.")

    # Get the dimensioxns of the (possibly resized) image
    height, width = image.shape[:2]

    # Define margins as a percentage of the image dimensions
    margin_percentage = 0.05  # 5% margin on each side
    margin_x = int(width * margin_percentage)
    margin_y = int(height * margin_percentage)

    # Calculate rectangle coordinates
    rect_x = margin_x
    rect_y = margin_y
    rect_w = width - 2 * margin_x
    rect_h = height - 2 * margin_y

    # The rectangle for GrabCut (x, y, width, height)
    rect = (rect_x, rect_y, rect_w, rect_h)
    print(f"Rectangle coordinates: {rect}")

    # Create an initial mask
    mask = np.zeros((height, width), np.uint8)

    # Define the background and foreground models (used internally by GrabCut)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Number of iterations for the GrabCut algorithm
    iterations = 5

    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask where the foreground is 0 (black) and background is 255 (white)
    mask_output = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 0, 255).astype('uint8')

    # If the image was resized, resize the mask back to original size
    if image_resized:
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]
        mask_output = cv2.resize(mask_output, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Save the black mask image
    cv2.imwrite(output_path, mask_output)
    print(f"Mask saved as '{output_path}'.")

    # Optionally, display the result
    if display_result:
        cv2.imshow('Black Mask', mask_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True

# Example usage:
if __name__ == "__main__":
    create_black_mask('beer.jpg', 'campus_mask3.png', display_result=True)