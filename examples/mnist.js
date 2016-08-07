'use strict';

var Promise = require('bluebird');

var _ = require('lodash');

var jsmind = require('../dist');

var net = new jsmind.Network([
  new jsmind.layers.ReLULayer(784, 100, {pDropout: 0.5}),
  new jsmind.layers.ReLULayer(100, 100, {pDropout: 0.5}),
  new jsmind.layers.SoftmaxLayer(100, 10, {pDropout: 0.5})
]);

Promise.all([
  jsmind.MnistLoader.loadTrainingDataWrapper(),
  jsmind.MnistLoader.loadValidationDataWrapper(),
  jsmind.MnistLoader.loadTestDataWrapper()
]).spread(function(trainingData, validationData, testData) {
  var testInput, prediction;

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

  testInput = _.unzip(testData)[0];
  prediction = net.predict(testInput);
  console.log('prediction:' + prediction.toString());
});
