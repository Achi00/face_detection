from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import requests
from io import BytesIO
import base64

# Initialize the Flask application
flask_app = Flask(__name__)

# Initialize the FaceAnalysis application
face_analysis_app = FaceAnalysis(name='buffalo_l')
face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

@flask_app.route('/swap-face', methods=['POST'])
def swap_face():
    try:
        # Read source (user's face) and target (generated) images from the request
        source_file = request.files['user_image']  # This should be the user's face
        target_image_url = request.form['generated_image_url']  # URL of the generated image

        # Convert the source image to OpenCV format
        source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Download and convert the target image from the URL
        response = requests.get(target_image_url)
        target_img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        target_img = cv2.imdecode(target_img_array, cv2.IMREAD_COLOR)

        # Detect faces
        source_faces = face_analysis_app.get(source_img)
        target_faces = face_analysis_app.get(target_img)
        if len(source_faces) != 1 or len(target_faces) != 1:
            raise Exception("Each image should have exactly one face.")

        # Initialize the face swapper model
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

        # Swap the face from source image onto the target image
        res = target_img.copy()
        res = swapper.get(res, target_faces[0], source_faces[0], paste_back=True)

        # Convert the result to base64 to send as JSON
        _, buffer = cv2.imencode('.jpg', res)
        res_base64 = base64.b64encode(buffer).decode()

        return jsonify({'result_image': res_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=5000)
