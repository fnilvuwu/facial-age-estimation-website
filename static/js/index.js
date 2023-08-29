const videoElement = document.getElementById('cam_input');
const canvasElement = document.getElementById('canvas_output');
const canvasRoi = document.getElementById('canvas_roi');
const canvasCtx = canvasElement.getContext('2d');
const roiCtx = canvasRoi.getContext('2d');
const noFaceFoundElement = document.getElementById('no-face-found');

const drawingUtils = window;
let tfliteModel;
let isModelLoaded = false;

async function loadModel() {
    try {
        tfliteModel = await tf.loadLayersModel("./static/model/uint8/model.json");
        isModelLoaded = true;
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load the model:", error);
    }
}

async function start() {
    await loadModel();
}

start();

function openCvReady() {
    cv['onRuntimeInitialized'] = () => {
        function onResults(results) {
            try {
                // Draw the overlays.
                canvasCtx.save();
                roiCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);
                canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

                if (results.detections.length > 0) {
                    drawingUtils.drawRectangle(
                        canvasCtx, results.detections[0].boundingBox,
                        { color: 'blue', lineWidth: 4, fillColor: '#00000000' });
                    let width = results.detections[0].boundingBox.width * canvasElement.width;
                    let height = results.detections[0].boundingBox.height * canvasElement.height;
                    let sx = results.detections[0].boundingBox.xCenter * canvasElement.width - (width / 2);
                    let sy = results.detections[0].boundingBox.yCenter * canvasElement.height - (height / 2);
                    let center = sx + (width / 2);

                    let imgData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                    let gray_roi = cv.matFromImageData(imgData);
                    let rect = new cv.Rect(sx, sy, width, height);
                    gray_roi = gray_roi.roi(rect);

                    cv.imshow('canvas_roi', gray_roi);

                    canvasCtx.font = "50px Arial";
                    canvasCtx.fillStyle = "red";
                    canvasCtx.textAlign = "center";

                    if (tfliteModel && isModelLoaded) {
                        // Perform the prediction
                        const outputTensor = tf.tidy(() => {
                            // Transform the image data into Array pixels.
                            let img = tf.browser.fromPixels(canvasRoi);

                            // Resize the image to [224, 224]
                            img = tf.image.resizeBilinear(img, [224, 224]);

                            // Normalize and expand dimensions of image pixels by 0 axis
                            img = tf.div(tf.expandDims(img, 0), 255);

                            // Predict the emotions.
                            let outputTensor = tfliteModel.predict(img).arraySync();
                            return outputTensor;
                        });

                        // Round the prediction
                        let predict = Math.round(outputTensor[0]);

                        if (isNaN(predict) || predict === undefined) {
                            canvasCtx.fillText("Age : Not available", center, sy - 50);
                        } else {
                            console.log(predict);
                            canvasCtx.fillText("Age : " + predict + " y.o.", center, sy - 50);
                        }

                        // Hide the "No face found" message
                        noFaceFoundElement.style.display = 'none';
                    } else {
                        canvasCtx.fillText("Loading the model", center, sy - 50);
                    }
                } else {
                    // No face detected
                    roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);
                    noFaceFoundElement.style.display = 'block';
                }
                canvasCtx.restore();
                roiCtx.restore();
            } catch (err) {
                console.log(err.message);
            }
        }

        const faceDetection = new FaceDetection({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
            }
        });

        faceDetection.setOptions({
            selfieMode: true,
            model: 'short',
            minDetectionConfidence: 0.1
        });

        faceDetection.onResults(onResults);

        const camera = new Camera(videoElement, {
            onFrame: async () => {
                await faceDetection.send({ image: videoElement });
            },
            width: 854,
            height: 480
        });

        camera.start();
    }
}

// Call the openCvReady function after start to ensure the model is loaded
start().then(() => {
    openCvReady();
});
