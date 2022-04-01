#!/usr/bin/env node
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const {loadTFLiteModel} = require('tfjs-tflite-node');
const {CoralDelegate} = require('coral-tflite-delegate');
const tf = require('@tensorflow/tfjs');
const jpeg = require('jpeg-js');
const fs = require('fs');

const modelPath = './mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite';

async function verify() {
  const model = await loadTFLiteModel(modelPath, {
    delegates: [new CoralDelegate()],
  });

  // Load the input image of a parrot.
  const parrotJpeg = jpeg.decode(
      fs.readFileSync('./parrot-small.jpg'));

  // Create an RGB array of the parrot's pixels (no Alpha channel).
  const {width, height, data} = parrotJpeg;
  const parrotRGB = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const i3 = i * 3;
    const i4 = i * 4;
    parrotRGB[i3] = data[i4];
    parrotRGB[i3 + 1] = data[i4 + 1];
    parrotRGB[i3 + 2] = data[i4 + 2];
  }

  parrot = tf.tensor(parrotRGB, [1, 224, 224, 3]);
  labels = fs.readFileSync('./inat_bird_labels.txt', 'utf-8')
      .split(/\r?\n/);

  const prediction = model.predict(parrot);
  const argmax = tf.argMax(prediction, 1);
  const label = labels[argmax.dataSync()[0]];
  const expectedLabel = 'Ara macao (Scarlet Macaw)';
  console.log('\n\n');
  if (label === expectedLabel) {
    console.log(`Coral device returned correct classification ${label}`);
  }
  else {
    throw new Error(`Coral device returned incorrect classification ${label}. `
                    + `Expected ${expectedLabel}`);
  }
}

verify();
