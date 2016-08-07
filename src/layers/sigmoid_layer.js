'use strict';

const FullyConnectedLayer = require('./fully_connected_layer');

class SigmoidLayer extends FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    opts.activationFn = 'sigmoid';
    super(nIn, nOut, opts);
  }

  costDelta(y) {
    return this.outputDropout.minus(y);
  }
}

module.exports = SigmoidLayer;
