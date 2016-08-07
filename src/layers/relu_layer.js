'use strict';

let FullyConnectedLayer = require('./fully_connected_layer');

class ReLULayer extends FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    opts.activationFn = 'relu';
    super(nIn, nOut, opts);
  }

  costDelta(y) {
    return this.outputDropout.eleMap(v => {
      return (v > 0) ? 1 : 0;
    });
  }
}

module.exports = ReLULayer;
