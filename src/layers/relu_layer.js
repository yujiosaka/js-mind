'use strict';

import FullyConnectedLayer from './fully_connected_layer';

export default class ReLULayer extends FullyConnectedLayer {
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
