"use strict";

var Promise = require("bluebird");

var jsmind = require("../dist");

var MnistLoader = jsmind.MnistLoader;
Promise.all([
  MnistLoader.loadTrainingDataWrapper(),
  MnistLoader.loadValidationDataWrapper(),
  MnistLoader.loadTestDataWrapper()
]).spread(function(trainingData, validationData, testData) {
  var net = new jsmind.Network([
    new jsmind.layers.FullyConnectedLayer(784, 100, {pDropout: 0.5}),
    new jsmind.layers.FullyConnectedLayer(100, 100, {pDropout: 0.5}),
    new jsmind.layers.FullyConnectedLayer(100, 10, {pDropout: 0.5})
  ]);

  net.SGD(trainingData, 60, 10, 0.1, {
    validationData: validationData,
    testData: testData,
    lmbda: 0.1
  });
});
