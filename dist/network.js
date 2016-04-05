"use strict";

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var _ = require("lodash");
var linearAlgebra = require("linear-algebra")();
var Matrix = linearAlgebra.Matrix;
var lib = require("./lib");

var Network = function () {
  function Network(layers) {
    _classCallCheck(this, Network);

    this.layers = layers;
  }

  _createClass(Network, [{
    key: "SGD",
    value: function SGD(trainingData, epochs, miniBatchSize, eta) {
      var opts = arguments.length <= 4 || arguments[4] === undefined ? {} : arguments[4];

      opts.validationData || (opts.validationData = null);
      opts.testData || (opts.testData = null);
      opts.lmbda || (opts.lmbda = 0);
      var bestValidationAccuracy = 0;
      var testAccuracy = null;

      var _iteratorNormalCompletion = true;
      var _didIteratorError = false;
      var _iteratorError = undefined;

      try {
        for (var _iterator = function () {
          var results = [];

          for (var i = 0; i < epochs; i++) {
            results.push(i);
          }

          return results;
        }.apply(this)[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
          var j = _step.value;

          trainingData = _.shuffle(trainingData);
          var miniBatches = this.createMiniBatches(trainingData, miniBatchSize);

          var _iteratorNormalCompletion2 = true;
          var _didIteratorError2 = false;
          var _iteratorError2 = undefined;

          try {
            for (var _iterator2 = miniBatches.entries()[Symbol.iterator](), _step2; !(_iteratorNormalCompletion2 = (_step2 = _iterator2.next()).done); _iteratorNormalCompletion2 = true) {
              var _step2$value = _slicedToArray(_step2.value, 2);

              var i = _step2$value[0];
              var miniBatch = _step2$value[1];

              var iteration = trainingData.length / miniBatchSize * j + i;
              iteration % 1000 === 0 ? console.log("Training mini-batch number " + iteration) : undefined;
              this.updateMiniBatch(miniBatch, eta, opts.lmbda, trainingData.length);
            }
          } catch (err) {
            _didIteratorError2 = true;
            _iteratorError2 = err;
          } finally {
            try {
              if (!_iteratorNormalCompletion2 && _iterator2.return) {
                _iterator2.return();
              }
            } finally {
              if (_didIteratorError2) {
                throw _iteratorError2;
              }
            }
          }

          if (opts.validationData) {
            var validationAccuracy = this.accuracy(opts.validationData);
            console.log("Epoch " + j + ": validation accuracy " + validationAccuracy);

            if (validationAccuracy >= bestValidationAccuracy) {
              console.log("This is the best validation accuracy to date.");
              bestValidationAccuracy = validationAccuracy;

              if (opts.testData) {
                testAccuracy = this.accuracy(opts.testData);
                console.log("The corresponding test accuracy " + testAccuracy);
              }
            }
          } else if (opts.testData) {
            testAccuracy = this.accuracy(opts.testData);
            console.log("Epoch " + j + ": test accuracy " + testAccuracy);
          }
        }
      } catch (err) {
        _didIteratorError = true;
        _iteratorError = err;
      } finally {
        try {
          if (!_iteratorNormalCompletion && _iterator.return) {
            _iterator.return();
          }
        } finally {
          if (_didIteratorError) {
            throw _iteratorError;
          }
        }
      }

      console.log("Finished training network.");

      if (opts.validationData) {
        console.log("Best validation accuracy " + bestValidationAccuracy);
        return opts.testData ? console.log("Corresponding test accuracy " + testAccuracy) : undefined;
      }
    }
  }, {
    key: "createMiniBatches",
    value: function createMiniBatches(trainingData, miniBatchSize) {
      return function () {
        var results = [];

        for (var i = 0, ref = trainingData.length; i < ref; i++) {
          results.push(i);
        }

        return results;
      }.apply(this).filter(function (_, _i) {
        return _i === 0 || _i % (miniBatchSize + 1) === 0;
      }).map(function (k) {
        return trainingData.slice(k, k + miniBatchSize);
      });
    }
  }, {
    key: "updateMiniBatch",
    value: function updateMiniBatch(miniBatch, eta, lmbda, n) {
      var _this = this;

      var x = new Matrix(miniBatch.map(function (_ref) {
        var _ref2 = _slicedToArray(_ref, 2);

        var _x = _ref2[0];
        var _y = _ref2[1];
        return _x.ravel();
      })).trans();
      var y = new Matrix(miniBatch.map(function (_ref3) {
        var _ref4 = _slicedToArray(_ref3, 2);

        var _x = _ref4[0];
        var _y = _ref4[1];
        return _y.ravel();
      })).trans();

      this.train(x, miniBatch.length);
      this.backprop(y);

      return function () {
        var _iteratorNormalCompletion3 = true;
        var _didIteratorError3 = false;
        var _iteratorError3 = undefined;

        try {
          for (var _iterator3 = _this.layers[Symbol.iterator](), _step3; !(_iteratorNormalCompletion3 = (_step3 = _iterator3.next()).done); _iteratorNormalCompletion3 = true) {
            var layer = _step3.value;

            layer.w = layer.w.mulEach(1 - eta * (lmbda / n)).minus(layer.nw.mulEach(eta / miniBatch.length));
            layer.b = layer.b.minus(layer.nb.mulEach(eta / miniBatch.length));
          }
        } catch (err) {
          _didIteratorError3 = true;
          _iteratorError3 = err;
        } finally {
          try {
            if (!_iteratorNormalCompletion3 && _iterator3.return) {
              _iterator3.return();
            }
          } finally {
            if (_didIteratorError3) {
              throw _iteratorError3;
            }
          }
        }
      }();
    }
  }, {
    key: "train",
    value: function train(x, miniBatchSize) {
      var _this2 = this;

      var initLayer = this.layers[0];
      initLayer.setInput(x, x, miniBatchSize);

      return function () {
        var _iteratorNormalCompletion4 = true;
        var _didIteratorError4 = false;
        var _iteratorError4 = undefined;

        try {
          for (var _iterator4 = function () {
            var results = [];

            for (var i = 1, ref = this.layers.length; i < ref; i++) {
              results.push(i);
            }

            return results;
          }.apply(_this2)[Symbol.iterator](), _step4; !(_iteratorNormalCompletion4 = (_step4 = _iterator4.next()).done); _iteratorNormalCompletion4 = true) {
            var j = _step4.value;

            var prevLayer = _this2.layers[j - 1];
            var layer = _this2.layers[j];
            layer.setInput(prevLayer.output, prevLayer.outputDropout, miniBatchSize);
          }
        } catch (err) {
          _didIteratorError4 = true;
          _iteratorError4 = err;
        } finally {
          try {
            if (!_iteratorNormalCompletion4 && _iterator4.return) {
              _iterator4.return();
            }
          } finally {
            if (_didIteratorError4) {
              throw _iteratorError4;
            }
          }
        }
      }();
    }
  }, {
    key: "backprop",
    value: function backprop(y) {
      var _this3 = this;

      var lastLayer = this.layers[this.layers.length - 1];
      var delta = lastLayer.costDelta(y);
      lastLayer.update(delta);

      return function () {
        var _iteratorNormalCompletion5 = true;
        var _didIteratorError5 = false;
        var _iteratorError5 = undefined;

        try {
          for (var _iterator5 = function () {
            var results = [];

            for (var i = 2, ref = this.layers.length + 1; 2 <= ref ? i < ref : i > ref; 2 <= ref ? i++ : i--) {
              results.push(i);
            }

            return results;
          }.apply(_this3)[Symbol.iterator](), _step5; !(_iteratorNormalCompletion5 = (_step5 = _iterator5.next()).done); _iteratorNormalCompletion5 = true) {
            var l = _step5.value;

            var followinglayer = _this3.layers[_this3.layers.length - l + 1];
            var layer = _this3.layers[_this3.layers.length - l];
            delta = followinglayer.w.trans().dot(delta).mul(lib.sigmoidPrime(layer.z));
            layer.update(delta);
          }
        } catch (err) {
          _didIteratorError5 = true;
          _iteratorError5 = err;
        } finally {
          try {
            if (!_iteratorNormalCompletion5 && _iterator5.return) {
              _iterator5.return();
            }
          } finally {
            if (_didIteratorError5) {
              throw _iteratorError5;
            }
          }
        }
      }();
    }
  }, {
    key: "accuracy",
    value: function accuracy(data) {
      var _this4 = this;

      return _.mean(data.map(function (_ref5) {
        var _ref6 = _slicedToArray(_ref5, 2);

        var x = _ref6[0];
        var y = _ref6[1];
        return _this4.feedforward(x).accuracy(y);
      }));
    }
  }, {
    key: "feedforward",
    value: function feedforward(a) {
      this.train(a, 1);
      return this.layers[this.layers.length - 1];
    }
  }, {
    key: "test",
    value: function test(data) {
      return this.accuracy(data);
    }
  }, {
    key: "predict",
    value: function predict(inputs) {
      var _this5 = this;

      return inputs.map(function (x) {
        return _this5.feedforward(x).yOut;
      });
    }
  }]);

  return Network;
}();

module.exports = Network;