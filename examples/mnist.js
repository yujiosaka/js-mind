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
  jsmind.MnistLoader.loadValidationDataWrapper()
]).spread(function(trainingData, validationData) {
  net.SGD(
    trainingData,
    60, // epochs
    10, // miniBatchSize
    0.03 // eta
  , {
    validationData: validationData,
    lmbda: 0.1
  });
}).then(function() {
  return jsmind.MnistLoader.loadTestDataWrapper();
}).then(function(testData) {
  var testInput, prediction, accuracy;
  testInput = _.unzip(testData)[0];
  accuracy = net.accuracy(testData);
  prediction = net.predict(testInput);
  console.log('Test accuracy ' + accuracy);
  console.log('Test prediction ' + prediction.toString());
});
