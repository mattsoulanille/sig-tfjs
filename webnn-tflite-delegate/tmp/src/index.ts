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

import * as path from 'path';
import {getLibPaths, getPlatform} from './utils';
import type {TFLiteDelegatePlugin} from '@tensorflow/tfjs-tflite-node';

const libPaths = getLibPaths('webnn_external_delegate_obj', path.join(__dirname, '../cc_lib'));

export class WebNNDelegate implements TFLiteDelegatePlugin {
  readonly name: 'WebNNDelegate';
  readonly tfliteVersion: '2.7';
  readonly node: TFLiteDelegatePlugin['node'];

  constructor(readonly options: [string, string][] = []) {
    const platform = getPlatform();
    const libPath = libPaths.get(platform);
    if (!libPath) {
      throw new Error(`Platform ${platform} is not supported`);
    }
    // Support 'webnn_device' option: (0:default, 1:gpu, 2:cpu)
    this.options = options;
    this.node = {
      path: libPath,
    };
  }
}
