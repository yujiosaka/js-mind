'use strict';

const linearAlgebra = require('linear-algebra')();
const { Matrix } = linearAlgebra;

/**
 * Derivative of the sigmoid function.
 * @param {Matrix} z
 * @return {Matrix} converted matrix with the same nRow, nCol
 */
export function sigmoidPrime(z) {
  return z.sigmoid().mul(z.sigmoid().mulEach(-1).plusEach(1));
};

/**
 * Randomly drop values from a layer.
 * @param {Matrix} layer
 * @param {number} pDropout probability to drop out value for each element
 * @return {Matrix} converted matrix with the same nRow, nCol
 */
export function dropoutLayer(layer, pDropout) {
  return layer.eleMap(function(elem) {
    return (Math.random() < pDropout ? 0 : elem);
  });
};

/**
 * Return a sample from a normal distribution.
 * @param {number} mu mean
 * @param {number} sigma sd (must be greater than 0)
 * @return {number} sample
 */
export function norm(mu, sigma) {
  const a = 1 - Math.random();
  const b = 1 - Math.random();
  const c = Math.sqrt(-2 * Math.log(a));
  if (0.5 - Math.random() > 0) {
    return c * Math.sin(Math.PI * 2 * b) * sigma + mu;
  } else {
    return c * Math.cos(Math.PI * 2 * b) * sigma + mu;
  }
}

/**
 * Return a matrix, all of whose element are sampled from the standard normal distribution.
 * see http://d.hatena.ne.jp/iroiro123/20111210/1323515616
 *
 * @param {number} rows the number of rows
 * @param {number} cols the number of cols
 * @return {Matrix} random matrix
 */
export function randn(rows, cols) {
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    result[i] = []
    for (let j = 0; j < cols; j++) {
      result[i][j] = norm(0, 1)
    }
  }
  return new Matrix(result);
}
