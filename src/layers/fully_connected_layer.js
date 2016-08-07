'use strict';

let linearAlgebra = require('linear-algebra')();
let Matrix = linearAlgebra.Matrix;
let lib = require('../lib');

class FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.pDropout = opts.pDropout || (opts.pDropout = 0);
    this.activationFn = opts.activationFn;
    this.w = lib.randn(this.nOut, this.nIn).mulEach(1 / Math.sqrt(this.nIn));
    this.b = lib.randn(this.nOut, 1);
  }

  setInput(input, inputDropout, miniBatchSize) {
    let bMask = new Matrix(this.b.ravel().map(v => {
      let results = [];
      for (let i = 0; i < miniBatchSize; i++) {
        results.push(v);
      }
      return results;
    }));
    let axis = 0;
    this.input = input;
    this.output = this.w.dot(input).mulEach(1 - this.pDropout).plus(bMask)[this.activationFn](axis);
    this.yOut = this.output.getArgMax();
    this.inputDropout = lib.dropoutLayer(inputDropout, this.pDropout);
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

module.exports = FullyConnectedLayer;
