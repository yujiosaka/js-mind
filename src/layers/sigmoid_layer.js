'use strict';

import FullyConnectedLayer from './fully_connected_layer';

export default class SigmoidLayer extends FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    opts.activationFn = 'sigmoid';
    super(nIn, nOut, opts);
  }

  costDelta(y) {
    return this.outputDropout.minus(y);
  }
}
