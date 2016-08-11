'use strict';

import _ from 'lodash';
import fs from 'fs';
import Promise from 'bluebird';
import linearAlgebra from 'linear-algebra';

import layers from './layers';

const { Matrix } = linearAlgebra();

const ENCODING = 'utf8';

Promise.promisifyAll(fs);

class Network {
  constructor(layers) {
    this.layers = layers;
  }

  SGD(trainingData, epochs, miniBatchSize, eta, opts = {}) {
    opts.lmbda || (opts.lmbda = 0);
    let bestValidationAccuracy = 0;
    for (let j = 0; j < epochs; j++) {
      trainingData = _.shuffle(trainingData);
      let miniBatches = this.createMiniBatches(trainingData, miniBatchSize);
      for (let i = 0; i < miniBatches.length; i++) {
        let miniBatch = miniBatches[i];
        let iteration = trainingData.length / miniBatchSize * j + i;
        if (iteration % 1000 === 0) {
          console.log(`Training mini-batch number ${iteration}`);
        }
        this.updateMiniBatch(miniBatch, eta, opts.lmbda, trainingData.length);
      }
      if (opts.validationData) {
        let validationAccuracy = this.accuracy(opts.validationData);
        console.log(`Epoch ${j}: validation accuracy ${validationAccuracy}`);
        if (validationAccuracy >= bestValidationAccuracy) {
          console.log('This is the best validation accuracy to date.');
          bestValidationAccuracy = validationAccuracy;
        }
      }
    }
    console.log('Finished training network.');
    if (opts.validationData) {
      console.log(`Best validation accuracy ${bestValidationAccuracy}`);
    }
  }

  createMiniBatches(trainingData, miniBatchSize) {
    let results = [];
    for (let k = 0; k < trainingData.length; k += miniBatchSize) {
      results.push(trainingData.slice(k, k + miniBatchSize));
    }
    return results;
  }

  updateMiniBatch(miniBatch, eta, lmbda, n) {
    let x = new Matrix(miniBatch.map(([_x, _y]) => { return _x.ravel();})).trans();
    let y = new Matrix(miniBatch.map(([_x, _y]) => { return _y.ravel();})).trans();
    this.train(x, miniBatch.length);
    this.backprop(y);
    for (let i = 0; i < this.layers.length; i++) {
      let layer = this.layers[i];
      // l2 regularization
      layer.w = layer.w.mulEach(1 - eta * (lmbda / n)).minus((layer.nw.mulEach(eta / miniBatch.length)));
      layer.b = layer.b.minus(layer.nb.mulEach(eta / miniBatch.length));
    }
  }

  train(x, miniBatchSize) {
    let initLayer = this.layers[0];
    initLayer.setInput(x, x, miniBatchSize);
    for (let j = 1; j < this.layers.length; j++) {
      let prevLayer = this.layers[j - 1];
      let layer = this.layers[j];
      layer.setInput(prevLayer.output, prevLayer.outputDropout, miniBatchSize);
    }
  }

  backprop(y) {
    let lastLayer = this.layers[this.layers.length - 1];
    let delta = lastLayer.costDelta(y);
    lastLayer.update(delta);
    for (let l = 2; l <= this.layers.length; l++) {
      let followinglayer = this.layers[this.layers.length - l + 1];
      let layer = this.layers[this.layers.length - l];
      delta = followinglayer.w.trans().dot(delta).mul(layer.costDelta(y));
      layer.update(delta);
    }
  }

  accuracy(data) {
    return _.mean(data.map(([x, y]) => { return this.feedforward(x).accuracy(y); }));
  }

  feedforward(a) {
    this.train(a, 1);
    return this.layers[this.layers.length - 1];
  }

  predict(inputs) {
    return inputs.map(x => { return this.feedforward(x).yOut; });
  }

  save(file) {
    let json = JSON.stringify(this.layers.map(layer => {
      return layer.dump();
    }));
    return fs.writeFileAsync(file, json, ENCODING);
  }

  static load(file) {
    return fs.readFileAsync(file, ENCODING).then(json => {
      return new Network(JSON.parse(json).map(data => {
        return layers[data.className].load(data.className, data.properties);
      }));
    });
  }
}

module.exports = Network;
