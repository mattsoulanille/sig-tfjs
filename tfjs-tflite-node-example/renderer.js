const tf = require('@tensorflow/tfjs');
const {loadTFLiteModel} = require('tfjs-tflite-node');
const fs = require('fs');
const {CoralDelegate} = require('coral-tflite-delegate');
const Stats = require('stats.js');

tf.setBackend('cpu');

async function getWebcam() {
  const webcam = document.createElement('video');
  webcam.width = 224;
  webcam.height = 224;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
        mandatory: {
          minHeight: webcam.height,
          minWidth: webcam.width,
          maxHeight: webcam.height,
          maxWidth: webcam.width,
        }
    }
  });
  // Flip the image so the video looks like a mirror.
  webcam.style.transform = 'scaleX(-1)';
  webcam.srcObject = stream;
  webcam.play();
  return webcam;
}

async function captureFrame(tensorCam) {
  const imageTensor = await tensorCam.capture();

  const result = tf.tidy(() => {
    //const resized = imageTensor.resizeBilinear([224, 224]);
    const expanded = tf.expandDims(imageTensor, 0);
    return expanded;
  });

  imageTensor.dispose();
  return result;
}

function convertToFloat(image) {
  // https://github.com/googlecreativelab/teachablemachine-community/blob/master/libraries/image/src/utils/tf.ts#L38
  return tf.tidy(() => {
    return image.toFloat()
        .div(tf.scalar(127))
        .sub(tf.scalar(1));
  });
}

function convertToUint(image) {
  return tf.tidy(() => {
    return tf.cast(image, 'int32');
  });
}

async function loadModel() {
  const model = await loadTFLiteModel('./model/model_unquant.tflite');
  const classes = fs.readFileSync('./model/labels.txt', 'utf8').split('\n');
  return [model, classes];
}

async function loadCoralModel() {
  const model = await loadTFLiteModel('./coral_model/model_edgetpu.tflite', {
    delegates: [new CoralDelegate()],
  });
  const classes = fs.readFileSync('./coral_model/labels.txt', 'utf8')
        .split('\n');
  return [model, classes];
}

async function main() {
  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const webcam = await getWebcam();
  document.body.appendChild(webcam);

  let useCoral = true;
  const toggleCoralButton = document.createElement('button');
  toggleCoralButton.innerHTML = useCoral ? 'Use CPU' : 'Use Coral';
  toggleCoralButton.addEventListener('click', () => {
    useCoral = !useCoral;
    toggleCoralButton.innerHTML = useCoral ? 'Use CPU' : 'Use Coral';
  });
  document.body.appendChild(toggleCoralButton);

  const tensorCam = await tf.data.webcam(webcam);

  // Add Classification header
  const header = document.createElement('h1');
  header.innerHTML = 'Classification: ';
  document.body.appendChild(header);

  const [model, classes] = await loadModel();
  const [coralModel] = await loadCoralModel();

  const resultList = document.createElement('ul');
  document.body.appendChild(resultList);

  function showPrediction(data, max) {
    resultList.innerHTML = ''; // Clear the list
    for (let i = 0; i < data.length; i++) {
      const percent = (data[i] * 100 / max).toFixed(1);
      const resultText = `${classes[i]}: ${percent}`;
      const result = document.createElement('li');
      result.innerHTML = resultText;
      resultList.appendChild(result);
    }
  }

  function predict(frame) {
    const imageTensor = convertToFloat(frame);
    const prediction = model.predict(imageTensor);
    const predictionData = prediction.dataSync();
    showPrediction(predictionData, 1);
  }

  function predictCoral(frame) {
    const imageTensor = convertToUint(frame);
    const prediction = coralModel.predict(imageTensor);
    const predictionData = prediction.dataSync();
    showPrediction(predictionData, 255);
  }

  async function run() {
    stats.begin();
    const frame = await captureFrame(tensorCam);
    if (useCoral) {
      predictCoral(frame);
    } else {
      predict(frame);
    }
    stats.end();
    requestAnimationFrame(run);
  }

  run();
}

main();
