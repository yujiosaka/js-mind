"use strict";

var _ = require("lodash");
var linearAlgebra = require("linear-algebra")();
var Matrix = linearAlgebra.Matrix;

var lib = require("./lib");

var loadTrainingData = function loadTrainingData() {
  var trainingInput = require("../data/training_input");
  var trainingOutput = require("../data/training_output");
  return [trainingInput, trainingOutput];
};

var loadValidationData = function loadValidationData() {
  var validationInput = require("../data/validation_input");
  var validationOutput = require("../data/validation_output");
  return [validationInput, validationOutput];
};

var loadTestData = function loadTestData() {
  var testInput = require("../data/test_input");
  var testOutput = require("../data/test_output");
  return [testInput, testOutput];
};

exports.loadTrainingDataWrapper = function () {
  var trD = loadTrainingData();
  var trainingInputs = trD[0].map(function (x) {
    return Matrix.reshape(x, 784, 1);
  });
  var trainingResults = trD[1].map(function (y) {
    return lib.vectorizedResult(y);
  });
  var trainingData = _.zip(trainingInputs, trainingResults);
  return trainingData;
};

exports.loadValidationDataWrapper = function () {
  var vaD = loadValidationData();
  var validationInputs = vaD[0].map(function (x) {
    return Matrix.reshape(x, 784, 1);
  });
  var validationData = _.zip(validationInputs, vaD[1]);
  return validationData;
};

exports.loadTestDataWrapper = function () {
  var teD = loadTestData();
  var testInputs = teD[0].map(function (x) {
    return Matrix.reshape(x, 784, 1);
  });
  var testData = _.zip(testInputs, teD[1]);
  return testData;
};