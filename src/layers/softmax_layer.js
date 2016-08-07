'use strict';

let FullyConnectedLayer = require('./fully_connected_layer');

class SoftmaxLayer extends FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    opts.activationFn = 'softmax';
    super(nIn, nOut, opts);
  }

  costDelta(y) {
    return this.outputDropout.minus(y);
  }
}

module.exports = SoftmaxLayer;
