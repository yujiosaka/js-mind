'use strict';

let _ = require('lodash');
let linearAlgebra = require('linear-algebra')();
let Matrix = linearAlgebra.Matrix;
let lib = require('./lib');

class Network {
  constructor(layers) {
    this.layers = layers;
  }

  SGD(trainingData, epochs, miniBatchSize, eta, opts = {}) {
    opts.validationData || (opts.validationData = null);
    opts.testData || (opts.testData = null);
    opts.lmbda || (opts.lmbda = 0);
    let bestValidationAccuracy = 0;
    let testAccuracy = null;

    for (let j of (function() {
        let results = [];

        for (let i = 0; i < epochs; i++) {
            results.push(i);
        }

        return results;
    }).apply(this)) {
      trainingData = _.shuffle(trainingData);
      let miniBatches = this.createMiniBatches(trainingData, miniBatchSize);

      for (let [i, miniBatch] of miniBatches.entries()) {
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

          if (opts.testData) {
            testAccuracy = this.accuracy(opts.testData);
            console.log(`The corresponding test accuracy ${testAccuracy}`);
          }
        }
      } else if (opts.testData) {
        testAccuracy = this.accuracy(opts.testData);
        console.log(`Epoch ${j}: test accuracy ${testAccuracy}`);
      }
    }

    console.log('Finished training network.');

    if (opts.validationData) {
      console.log(`Best validation accuracy ${bestValidationAccuracy}`);
      if (opts.testData) {
        console.log(`Corresponding test accuracy ${testAccuracy}`);
      }
    }
  }

  createMiniBatches(trainingData, miniBatchSize) {
    return ((function() {
        let results = [];

        for (let i = 0, ref = trainingData.length; i < ref; i++) {
            results.push(i);
        }

        return results;
    }).apply(this).filter((_, _i) => {
      return _i === 0 || _i % (miniBatchSize + 1) === 0;
    }).map(k => {
      return trainingData.slice(k, (k + miniBatchSize));
    }));
  }

  updateMiniBatch(miniBatch, eta, lmbda, n) {
    let x = new Matrix(miniBatch.map(([_x, _y]) => { return _x.ravel();})).trans();
    let y = new Matrix(miniBatch.map(([_x, _y]) => { return _y.ravel();})).trans();

    this.train(x, miniBatch.length);
    this.backprop(y);

    return (() => {
      for (let layer of this.layers) {
        layer.w = layer.w.mulEach(1 - eta * (lmbda / n)).minus((layer.nw.mulEach(eta / miniBatch.length)));
        layer.b = layer.b.minus(layer.nb.mulEach(eta / miniBatch.length));
      }
    })();
  }

  train(x, miniBatchSize) {
    let initLayer = this.layers[0];
    initLayer.setInput(x, x, miniBatchSize);

    return (() => {
      for (let j of (function() {
          let results = [];

          for (let i = 1, ref = this.layers.length; i < ref; i++) {
              results.push(i);
          }

          return results;
      }).apply(this)) {
        let prevLayer = this.layers[j - 1];
        let layer = this.layers[j];
        layer.setInput(prevLayer.output, prevLayer.outputDropout, miniBatchSize);
      }
    })();
  }

  backprop(y) {
    let lastLayer = this.layers[this.layers.length - 1];
    let delta = lastLayer.costDelta(y);
    lastLayer.update(delta);

    return (() => {
      for (let l of (function() {
          let results = [];

          for (let i = 2, ref = this.layers.length + 1; (2 <= ref ? i < ref : i > ref); (2 <= ref ? i++ : i--)) {
              results.push(i);
          }

          return results;
      }).apply(this)) {
        let followinglayer = this.layers[this.layers.length - l + 1];
        let layer = this.layers[this.layers.length - l];
        delta = followinglayer.w.trans().dot(delta).mul(lib.sigmoidPrime(layer.z));
        layer.update(delta);
      }
    })();
  }

  accuracy(data) {
    return _.mean(data.map(([x, y]) => { return this.feedforward(x).accuracy(y); }));
  }

  feedforward(a) {
    this.train(a, 1);
    return this.layers[this.layers.length - 1];
  }

  test(data) {
    return this.accuracy(data);
  }

  predict(inputs) {
    return inputs.map(x => { return this.feedforward(x).yOut; });
  }
}

module.exports = Network;
