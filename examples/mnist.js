'use strict';

var Promise = require('bluebird');

var jsmind = require('../dist');

var net = new jsmind.Network([
  new jsmind.layers.ReLULayer(784, 100),
  new jsmind.layers.SoftmaxLayer(100, 10)
]);

Promise.all([
  jsmind.MnistLoader.loadTrainingDataWrapper(),
  jsmind.MnistLoader.loadValidationDataWrapper(),
  jsmind.MnistLoader.loadTestDataWrapper()
]).spread(function(trainingData, validationData, testData) {
  net.SGD(
    trainingData,
    60, // epochs
    10, // miniBatchSize
    0.03 // eta
  , {
    validationData: validationData,
    testData: testData,
    lmbda: 0.1
  });
});
