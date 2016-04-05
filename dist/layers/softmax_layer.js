"use strict";

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var linearAlgebra = require("linear-algebra")();
var Matrix = linearAlgebra.Matrix;
var lib = require("../lib");

var SoftmaxLayer = function () {
  function SoftmaxLayer(nIn, nOut) {
    var opts = arguments.length <= 2 || arguments[2] === undefined ? {} : arguments[2];

    _classCallCheck(this, SoftmaxLayer);

    this.nIn = nIn;
    this.nOut = nOut;
    this.pDropout = opts.pDropout || (opts.pDropout = 0);
    this.w = Matrix.zeros(this.nOut, this.nIn);
    this.b = Matrix.zeros(this.nOut, 1);
  }

  _createClass(SoftmaxLayer, [{
    key: "setInput",
    value: function setInput(input, inputDropout, miniBatchSize) {
      var bMask = new Matrix(this.b.ravel().map(function (v) {
        var results = [];
        for (var i = 0; i < miniBatchSize; i++) {
          results.push(v);
        }
        return results;
      }));
      this.input = input;
      this.z = this.w.dot(input).mulEach(1 - this.pDropout).plus(bMask);
      this.output = this.z.softmax(0);
      this.yOut = this.output.getArgMax();
      this.inputDropout = lib.dropoutLayer(inputDropout, this.pDropout);
      return this.outputDropout = this.w.dot(this.inputDropout).plus(bMask).softmax(0);
    }
  }, {
    key: "accuracy",
    value: function accuracy(y) {
      return this.yOut === y;
    }
  }, {
    key: "costDelta",
    value: function costDelta(y) {
      return this.outputDropout.minus(y);
    }
  }, {
    key: "update",
    value: function update(delta) {
      this.nb = new Matrix(delta.getSum(1)).trans();
      return this.nw = delta.dot(this.input.trans());
    }
  }]);

  return SoftmaxLayer;
}();

module.exports = SoftmaxLayer;