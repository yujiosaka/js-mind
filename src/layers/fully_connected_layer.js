'use strict';

import linearAlgebra from 'linear-algebra';

import { randn, dropoutLayer } from '../lib';

const { Matrix } = linearAlgebra();

export default class FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    this.pDropout = opts.pDropout || (opts.pDropout = 0);
    this.activationFn = opts.activationFn;
    this.w = randn(nOut, nIn).mulEach(1 / Math.sqrt(nIn));
    this.b = randn(nOut, 1);
  }

  setInput(input, inputDropout, miniBatchSize) {
    const axis = 0;
    const bMask = new Matrix(this.b.ravel().map(v => {
      let results = [];
      for (let i = 0; i < miniBatchSize; i++) {
        results.push(v);
      }
      return results;
    }));
    this.input = input;
    this.output = this.w.dot(input).mulEach(1 - this.pDropout).plus(bMask)[this.activationFn](axis);
    this.yOut = this.output.getArgMax();
    this.inputDropout = dropoutLayer(inputDropout, this.pDropout);
    this.outputDropout = this.w.dot(this.inputDropout).plus(bMask)[this.activationFn](axis);
  }

  accuracy(y) {
    return this.yOut === y;
  }

  update(delta) {
    this.nb = new Matrix(delta.getSum(1)).trans();
    this.nw = delta.dot(this.input.trans());
  }
}
