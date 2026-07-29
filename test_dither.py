import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy.ndimage import binary_closing, binary_fill_holes, label

def test():
    IMG_PATH = "72bbc977-d086-4ce3-b9a0-5b253666002e.png"
    img = Image.open(IMG_PATH).convert("RGB")
    
    # Better crop: The user's face is in the upper middle.
    w, h = img.size
    crop_box = (w*0.2, h*0.1, w*0.8, h*0.7)
    img = img.crop(crop_box)
    img = img.resize((300, 340), Image.Resampling.LANCZOS)
    img_gray = img.convert("L")
    
    img_gray = ImageOps.autocontrast(img_gray, cutoff=1)
    img_gray = img_gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(1.3)
    
    img_gray.save("test_gray.png")
    
    arr = np.array(img_gray, dtype=float)
    
    # Let's try Otsu's thresholding or a simple percentile
    threshold = np.percentile(arr, 30) # Bottom 30% is background/suit
    mask = arr > threshold
    
    # Clean up mask
    mask = binary_closing(mask, structure=np.ones((5,5)))
    mask = binary_fill_holes(mask)
    
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save("test_mask.png")

    # Dither
    dots = []
    GRID_W, GRID_H = 300, 340
    for y in range(GRID_H):
        row_iter = range(GRID_W) if y % 2 == 0 else reversed(range(GRID_W))
        for x in row_iter:
            old_pixel = arr[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            
            if not mask[y, x]:
                new_pixel = 0
                arr[y, x] = 0
                
            if new_pixel == 255:
                dots.append((x, y))
                
            quant_error = old_pixel - new_pixel
            if y % 2 == 0:
                if x + 1 < GRID_W: arr[y, x + 1] += quant_error * 7 / 16
                if y + 1 < GRID_H:
                    if x - 1 >= 0: arr[y + 1, x - 1] += quant_error * 3 / 16
                    arr[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < GRID_W: arr[y + 1, x + 1] += quant_error * 1 / 16
            else:
                if x - 1 >= 0: arr[y, x - 1] += quant_error * 7 / 16
                if y + 1 < GRID_H:
                    if x + 1 < GRID_W: arr[y + 1, x + 1] += quant_error * 3 / 16
                    arr[y + 1, x] += quant_error * 5 / 16
                    if x - 1 >= 0: arr[y + 1, x - 1] += quant_error * 1 / 16
                    
    # Render dots
    out = Image.new("RGB", (300, 340), "black")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(out)
    for x, y in dots:
        draw.point((x, y), fill="white")
    out.save("test_dots_dark.png")
    
test()
