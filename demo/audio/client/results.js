/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';
import MediaStreamRecorder from 'msr';
import {ClientAPI} from 'federated-learning-client';
import {plotSpectrogram, plotSpectrum} from './spectral_plots';
import {loadAudioTransferLearningModel} from './model';
import {FrequencyListener} from './frequency_listener';
import {getNextLabel, labelNames} from './labels';

let serverURL = location.origin;
if (URLSearchParams) {
  const params = new URLSearchParams(location.search);
  if (params.get('server')) {
    serverURL = params.get('server');
  }
}

loadAudioTransferLearningModel().then(async (model) => {
  const clientAPI = new ClientAPI(model);
  await clientAPI.connect(serverURL);


  const data = clientAPI.getData();
  const models = clientAPI.getModels();

  const versions = sorted(models.map(m => m.version));
  const dataByVersionByClient = {};

  for (let i = 0; i < data.length; i++) {
    const version = data[i].modelVersion;
    const client = data[i].clientId
    if (!dataByVersion[version]) {
      dataByVersion[version] = {};
    }
    if (!dataByVersion[version][client]) {
      dataByVersion[version][client] = [];
    }
    dataByVersion[version][client].push(data[i]);
  }

  const clientScatter = {
    x: [],
    y: [],
    mode: 'markers',
    type: 'scatter'
  }

  const meanClientLine = {
    x: [],
    y: [],
    mode: 'lines',
    type: 'scatter'
  }

  for (let i = 0; i < versions.length; i++) {
    let totalAcrossClients = 0;
    let totalNumDataValues = 0;
    const v = versions[i];
    for (let c in dataByVersionByClient[v]) {
      let totalForThisClient = 0;
      const dataValues = dataByVersionByClient[v][c];
      dataValues.forEach(datum => {
        const loss = lossFor(datum);
        totalForThisClient += loss;
        totalAcrossClients += loss;
      });
      totalNumDataValues += dataValues.length;
      clientScatter.x.push(i);
      clientScatter.y.push(
          totalForThisClient / dataValues.length);
    }
    meanClientLine.x.push(i);
    meanClientLine.y.push(
       totalAcrossClients / totalNumDataValues);
  }

  Plotly.newPlot('results', [clientScatter, meanClientLine], {
    autosize: false,
    width: 480,
    height: 180,
    margin: {l: 30, r: 5, b: 30, t: 5, pad: 0},
  });
});

