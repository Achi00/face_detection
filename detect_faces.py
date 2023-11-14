import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

# Initialize the FaceAnalysis application
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load source image (with the face to be swapped)
source_img = cv2.imread('input.jpg')
if source_img is None:
    raise ValueError("Failed to load image 'input.png'.")

# Detect face in source image
source_faces = app.get(source_img)
if len(source_faces) != 1:
    raise Exception("The source image should have exactly one face.")

# Load target face image (face to swap onto)
target_img = cv2.imread('target.png')
if target_img is None:
    raise ValueError("Failed to load image 'target_face.jpg'.")

# Detect face in target image
target_faces = app.get(target_img)
if len(target_faces) != 1:
    raise Exception("The target image should have exactly one face.")

# Initialize the face swapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Swap the face from source image onto the target image
res = target_img.copy()
res = swapper.get(res, target_faces[0], source_faces[0], paste_back=True)

# Display the result
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
