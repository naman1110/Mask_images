#pip install torch torchvision transformers pillow opencv-python
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def create_black_mask(
    image_path,
    prompt,
    output_path='mask_output.png',
    threshold=0.5,
    invert_mask=True,
    return_mask=True,
    grow_mask=0,
    seed=None,
    base64_output=False,
    display_result=False
):
    """
    Creates a black mask of the specified object in an image using text-guided segmentation.

    Parameters:
    - image_path (str): Path to the input image.
    - prompt (str): Text prompt specifying the object to mask.
    - output_path (str): Path to save the output mask image.
    - threshold (float): Threshold to binarize the segmentation mask.
    - invert_mask (bool): Whether to invert the mask (object as black).
    - return_mask (bool): Whether to return the mask as a numpy array.
    - grow_mask (int): Number of pixels to dilate the mask.
    - seed (int): Random seed for reproducibility.
    - base64_output (bool): Whether to return the mask as a base64 string.
    - display_result (bool): Whether to display the resulting mask.

    Returns:
    - mask_output (numpy array or base64 string): The generated mask.
    """

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Initialize the processor and model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Preprocess the image and prompt
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits

    # Resize the prediction to the original image size
    pred_mask = torch.nn.functional.interpolate(
        preds.unsqueeze(1),
        size=image.size[::-1],  # PIL images are (width, height)
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Convert to numpy array and apply threshold
    mask = pred_mask.sigmoid().cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Optionally, grow the mask
    if grow_mask > 0:
        kernel = np.ones((grow_mask, grow_mask), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Invert the mask if needed
    if invert_mask:
        binary_mask = cv2.bitwise_not(binary_mask)

    # Save the mask if output_path is specified
    if output_path:
        cv2.imwrite(output_path, binary_mask)
        print(f"Mask saved as '{output_path}'.")

    # Optionally, display the result
    if display_result:
        cv2.imshow('Mask', binary_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the mask
    if base64_output:
        import base64
        from io import BytesIO

        pil_mask = Image.fromarray(binary_mask)
        buffered = BytesIO()
        pil_mask.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return mask_base64

    if return_mask:
        return binary_mask

    return None

# Example usage
if __name__ == "__main__":
    data = {
        "prompt": "bottle",
        "image": 'beer.jpg',
        "threshold": 0.2,
        "invert_mask": True,
        "return_mask": True,
        "grow_mask": 10,
        "seed": 468685,
        "base64": False
    }

    mask = create_black_mask(
        image_path=data["image"],
        prompt=data["prompt"],
        output_path='beer_mask.png',
        threshold=data["threshold"],
        invert_mask=data["invert_mask"],
        return_mask=data["return_mask"],
        grow_mask=data["grow_mask"],
        seed=data["seed"],
        base64_output=data["base64"],
        display_result=True
    )