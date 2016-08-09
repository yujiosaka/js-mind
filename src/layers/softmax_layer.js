'use strict';

import FullyConnectedLayer from './fully_connected_layer';

export default class SoftmaxLayer extends FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    opts.activationFn = 'softmax';
    super(nIn, nOut, opts);
  }

  costDelta(y) {
    return this.outputDropout.minus(y);
  }
}
