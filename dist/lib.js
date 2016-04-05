"use strict";

var linearAlgebra = require("linear-algebra")();
var Matrix = linearAlgebra.Matrix;

/*
 * Derivative of the sigmoid function.
 */
exports.sigmoidPrime = function (z) {
  return z.sigmoid().mul(z.sigmoid().mulEach(-1).plusEach(1));
};

/*
 * Return a 10-dimensional unit vector with a 1.0 in the j'th position
 * and zeroes elsewhere.  This is used to convert a digit (0...9)
 * into a corresponding desired output from the neural network.
 */
exports.vectorizedResult = function (j) {
  var e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (i) {
    return [0];
  });
  e[j] = [1];
  return new Matrix(e);
};

/*
 * Randomly drop values from a layer.
 */
exports.dropoutLayer = function (layer, pDropout) {
  return layer.eleMap(function (elem) {
    return Math.random() < pDropout ? 0 : elem;
  });
};

/*
 * Return a sample from a normal distribution.
 */
var norm = exports.norm = function (mu, sigma) {
  var a = 1 - Math.random();
  var b = 1 - Math.random();
  var c = Math.sqrt(-2 * Math.log(a));
  if (0.5 - Math.random() > 0) {
    return c * Math.sin(Math.PI * 2 * b) * sigma + mu;
  } else {
    return c * Math.cos(Math.PI * 2 * b) * sigma + mu;
  }
};

/*
 * Return samples from the standard normal distribution.
 * see http://d.hatena.ne.jp/iroiro123/20111210/1323515616
 */
exports.randn = function (rows, cols) {
  var result = new Array(rows);
  for (var i = 0; i < rows; i++) {
    result[i] = [];
    for (var j = 0; j < cols; j++) {
      result[i][j] = norm(0, 1);
    }
  }
  return result;
};