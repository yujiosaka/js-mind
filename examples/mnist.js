'use strict';

var Promise = require('bluebird');

var jsmind = require('../dist');

Promise.all([
  jsmind.MnistLoader.loadTrainingDataWrapper(),
  jsmind.MnistLoader.loadValidationDataWrapper(),
  jsmind.MnistLoader.loadTestDataWrapper()
]).spread(function(trainingData, validationData, testData) {
  var net = new jsmind.Network([
    new jsmind.layers.ReLULayer(784, 100),
    new jsmind.layers.SoftmaxLayer(100, 10)
  ]);
  net.SGD(trainingData, 60, 10, 0.03, {
    validationData: validationData,
    testData: testData,
    lmbda: 0.1
  });
});
