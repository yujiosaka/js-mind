'use strict';

let linearAlgebra = require('linear-algebra')();
let Matrix = linearAlgebra.Matrix;
let lib = require('../lib');

class FullyConnectedLayer {
  constructor(nIn, nOut, opts = {}) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.pDropout = opts.pDropout || (opts.pDropout = 0);
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
    this.input = input;
    this.z = this.w.dot(input).mulEach(1 - this.pDropout).plus(bMask);
    this.output = this.z.sigmoid();
    this.yOut = this.output.getArgMax();
    this.inputDropout = lib.dropoutLayer(inputDropout, this.pDropout);
    return this.outputDropout = this.w.dot(this.inputDropout).plus(bMask).sigmoid();
  }

  accuracy(y) {
    return this.yOut === y;
  }

  costDelta(y) {
    return this.outputDropout.minus(y).mul(lib.sigmoidPrime(this.z));
  }

  update(delta) {
    this.nb = new Matrix(delta.getSum(1)).trans();
    return this.nw = delta.dot(this.input.trans());
  }
}

module.exports = FullyConnectedLayer;
