from skimage.metrics import structural_similarity as ssim
from skimage import io, color, transform

# Load reference images
reference_images = []
for i in range(1, 11):
    img = io.imread('datasets/erica/erica' + str(i) + '.jpg')
    img_resized = transform.resize(img, (239, 235), mode='constant')
    img_gray = color.rgb2gray(img_resized)
    reference_images.append(img_gray)
    
# Load target images for dreambooth
target_images_dreambooth = []
for i in range(1, 7):
    img = io.imread('generated_samples/dreambooth/db' + str(i) + '.png')
    img_rgb = color.rgba2rgb(img)
    img_gray = color.rgb2gray(img_rgb)
    target_images_dreambooth.append(img_gray)

# Load target images for textual inversion
target_images_textual_inversion = []
for i in range(1, 8):
    img = io.imread('generated_samples/textual_inversion/ti' + str(i) + '.png')
    img_rgb = color.rgba2rgb(img)
    img_gray = color.rgb2gray(img_rgb)
    target_images_textual_inversion.append(img_gray)

# Calcualte SSIM values
ssim_dreambooth = []
for target in target_images_dreambooth:
    for ref in reference_images:
        ssim_value, _ = ssim(target, ref, full=True)
        ssim_dreambooth.append(ssim_value)
ssim_textual_inversion = []
for target in target_images_textual_inversion:
    for ref in reference_images:
        ssim_value, _ = ssim(target, ref, full=True)
        ssim_textual_inversion.append(ssim_value)

# Take the average
ssim_dreambooth_avg = sum(ssim_dreambooth) / len(ssim_dreambooth)
ssim_textual_inversion_avg = sum(ssim_textual_inversion) / len(ssim_textual_inversion)

print(f"SSIM Value Dreambooth: {ssim_dreambooth_avg}")
print(f"SSIM Value Textual Inversion: {ssim_textual_inversion_avg}")
