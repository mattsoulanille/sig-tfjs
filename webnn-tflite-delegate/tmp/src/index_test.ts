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

import {WebNNDelegate} from './index';
import {loadTFLiteModel} from '@tensorflow/tfjs-tflite-node'
import {TFLiteModel} from '@tensorflow/tfjs-tflite-node/dist/tflite_model';
import * as fs from 'fs';
import * as tfnode from '@tensorflow/tfjs-node';
import {Tensor} from '@tensorflow/tfjs-core';

describe('webnn delegate', () => {
  const modelPath = './test_model/mobilenetv2.tflite';
  let model: TFLiteModel;
  let wine: Tensor;
  let labels: string[];

  beforeEach(async () => {
    model = await loadTFLiteModel(modelPath, {
      // 'webnn_device' option: (0:default, 1:gpu, 2:cpu)
      delegates: [new WebNNDelegate([['webnn_device', '1']])],
    });
    // Pre-processing is referred from
    // https://github.com/tensorflow/tfjs-models/blob/master/mobilenet/src/index.ts

    // Load the input image of a wine.
    const wineJpeg = fs.readFileSync('./test_model/wine.jpeg');
    const img = tfnode.node.decodeJpeg(wineJpeg);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized: tfnode.Tensor3D = tfnode.add(tfnode.mul(tfnode.cast(
        img, 'float32'), 2 / 255.0), -1);
    // Resize the image if need.
    let resized = normalized;
    if (img.shape[0] !== 224 || img.shape[1] !== 224) {
      resized = tfnode.image.resizeBilinear(normalized, [224, 224], true);
    }
    // Reshape so we can pass it to predict.
    wine = tfnode.reshape(resized, [-1, 224, 224, 3]);
    labels = fs.readFileSync('./test_model/mobilenetv2_labels.txt', 'utf-8')
      .split(/\r?\n/);
  });

  it('runs a mobilenetv2 model', () => {
    const prediction = model.predict(wine);
    const slice = tfnode.slice(prediction as Tensor, [0, 1], [-1, 1000]);
    const argmax = tfnode.argMax(slice as Tensor, 1);
    const labelIndex = argmax.dataSync()[0];
    const label = labels[labelIndex];
    console.log('label:', label);
    console.log('score: ', slice.dataSync()[labelIndex]);
    expect(label).toEqual('wine bottle');
  });
});
