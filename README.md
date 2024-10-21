# Image-Processing
import os
import random
import numpy as np
from skimage import io
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from scipy.spatial.distance import euclidean
from scipy.stats import chisquare
import matplotlib.pyplot as plt


classes = {
    'animal': 'animal',
    'moon': 'moon',
    'nature': 'nature'
}
def load_selected_image(class_name, image_name):
    img_path = os.path.join(classes[class_name], image_name)
    image = io.imread(img_path)
    return image, img_path

def load_random_image(classes):
    img = os.listdir(classes)
    selected_image = random.choice(img)
    img_path = os.path.join(classes, selected_image)
    image = io.imread(img_path)
    return image, img_path

def extract_lbp_features(image, P=8):
    image_gray = rgb2gray(image)
    image_gray = (image_gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(image_gray, P, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist, lbp

first_class = 'animal'  # Specify the class name
first_image_name = 'animal 1.jpeg'  # Specify the image file name
first_image, first_image_path = load_selected_image(first_class, first_image_name)
first_image_lbp, first_image_lbp_img = extract_lbp_features(first_image)

S_classes = list(set(classes.keys()))  # Get classes excluding the first class
second_class = random.choice(S_classes)
random_image, random_image_path = load_random_image(classes[second_class])
random_image_lbp, random_image_lbp_img = extract_lbp_features(random_image)

distance = euclidean(first_image_lbp, random_image_lbp)
#distance = chisquare(first_image_lbp, random_image_lbp)

# Output
print(f"Texture of the first image from class '{first_class}': {first_image_lbp}")
print(f"Randomly selected image from class '{second_class}': {random_image_lbp}")
print(f"Distance between the first image and the randomly selected image: {distance:.4f}")

threshold = 0.1

if distance<threshold:
    classification = "similar"
else:
    classification = "different"

plt.figure(figsize=(15, 9))

plt.subplot(2, 2, 1)
plt.imshow(first_image)
plt.title('First Image from ' + first_class)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(random_image)
plt.title('Randomly Selected Image from ' + second_class)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(first_image_lbp_img, cmap='gray')
plt.title('LBP Image for First Image')
plt.axis('off')git init

plt.subplot(2, 2, 4)
plt.imshow(random_image_lbp_img, cmap='gray')
plt.title('LBP Image for Random Image')
plt.axis('off')

plt.suptitle('Results: The images are classified as ' + classification + '. Distance: ' + str(round(distance, 4)))
plt.tight_layout()
plt.show()
