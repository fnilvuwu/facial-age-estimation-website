const videoElement = document.getElementById('cam_input');
const canvasElement = document.getElementById('canvas_output');
const canvasRoi = document.getElementById('canvas_roi');
const canvasCtx = canvasElement.getContext('2d');
const roiCtx = canvasRoi.getContext('2d');

const drawingUtils = window;
const mpFaceDetection = window;
let tfliteModel;
let isModelLoaded = false;
let lastPredictionTime = 0;
let lastPredictedAge = null;
const predictionInterval = 500; // 5 seconds

// Load TensorFlow.js model
async function loadModel() {
    try {
        tfliteModel = await tf.loadLayersModel("./static/model/uint8/model.json");
        isModelLoaded = true;
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load the model:", error);
    }
}

// Set up camera using MediaDevices API
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;

        // Wait for the video metadata to load
        await new Promise((resolve, reject) => {
            videoElement.onloadedmetadata = () => {
                console.log("Camera metadata loaded.");
                resolve();
            };
            videoElement.onerror = (err) => {
                console.error("Error loading video element:", err);
                reject(err);
            };
        });

        await videoElement.play();
        console.log("Camera playback started successfully!");

        // Set canvas dimensions to match the video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

    } catch (error) {
        console.error("Failed to access the camera:", error);
        alert(`Unable to access the camera. Reason: ${error.message}`);
    }
}

// Initialize MediaPipe components
async function initializeMediaPipe() {
    await setupCamera();
    await loadModel();

    // Check if the video element is properly initialized
    if (!videoElement || !videoElement.srcObject) {
        console.error("Video element is not properly initialized.");
        alert("Video element is missing or not receiving the camera feed.");
        return;
    }

    console.log("Video element is ready and receiving camera feed.");

    function onResults(results) {
        console.log("Face detection results:", results); // Log results to check detection
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        if (results.detections.length > 0) {
            const detection = results.detections[0];
            drawingUtils.drawRectangle(canvasCtx, detection.boundingBox, { color: 'blue', lineWidth: 4, fillColor: '#00000000' });

            // Extract ROI
            const bbox = detection.boundingBox;
            const width = bbox.width * canvasElement.width;
            const height = bbox.height * canvasElement.height;
            const sx = bbox.xCenter * canvasElement.width - width / 2;
            const sy = bbox.yCenter * canvasElement.height - height / 2;

            const imgData = canvasCtx.getImageData(sx, sy, width, height);

            // Save the original ROI
            const originalRoiCanvas = document.createElement('canvas');
            originalRoiCanvas.width = width;
            originalRoiCanvas.height = height;
            const originalRoiCtx = originalRoiCanvas.getContext('2d');
            originalRoiCtx.putImageData(imgData, 0, 0);

            // Create an off-screen canvas to scale the ROI
            const offScreenCanvas = document.createElement('canvas');
            offScreenCanvas.width = 224;
            offScreenCanvas.height = 224;
            const offScreenCtx = offScreenCanvas.getContext('2d');
            offScreenCtx.drawImage(originalRoiCanvas, 0, 0, width, height, 0, 0, 224, 224);

            // Draw the scaled ROI on the canvasRoi
            roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);

            // Update canvas size dynamically if the image data size changes.
            canvasRoi.width = imgData.width;
            canvasRoi.height = imgData.height;

            // Draw the image data centered in the canvas.
            roiCtx.putImageData(imgData, 0, 0);


            // Preprocess original ROI and predict age
            const currentTime = Date.now();
            if (isModelLoaded && currentTime - lastPredictionTime > predictionInterval) {
                lastPredictionTime = currentTime;
                const img = tf.browser.fromPixels(originalRoiCanvas);
                const resizedImg = tf.image.resizeBilinear(img, [224, 224]);
                const normalizedImg = resizedImg.div(255).expandDims(0);
                const prediction = tfliteModel.predict(normalizedImg);
                lastPredictedAge = Math.round(prediction.dataSync()[0]);
                console.log("Predicted age:", lastPredictedAge);
            }

            // Display the predicted age if within the interval
            if (lastPredictedAge !== null && currentTime - lastPredictionTime <= predictionInterval) {
                canvasCtx.font = "20px Arial";
                canvasCtx.fillStyle = "red";
                canvasCtx.fillText(`Age: ${lastPredictedAge}`, sx, sy - 10);
            }
        } else {
            // Display "No face found" text in the center of the canvas
            canvasCtx.font = "30px Arial";
            canvasCtx.fillStyle = "red";
            canvasCtx.textAlign = "center";
            canvasCtx.fillText("No face found", canvasElement.width / 2, canvasElement.height / 2);
        }
        canvasCtx.restore();
    }

    const faceDetection = new mpFaceDetection.FaceDetection({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
        }
    });

    faceDetection.setOptions({
        selfieMode: true,
        model: 'short',
        minDetectionConfidence: 0.5
    });

    faceDetection.onResults(onResults);

    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await faceDetection.send({ image: videoElement });
        },
        width: 854,
        height: 480
    });

    camera.start().then(() => {
        console.log("MediaPipe camera started successfully!");
    }).catch(error => {
        console.error("Failed to start MediaPipe camera:", error);
    });
}

// Start the application
async function startApp() {
    try {
        await initializeMediaPipe();
    } catch (error) {
        console.error("Error during initialization:", error);
    }
}

// Start the application when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    startApp();
});