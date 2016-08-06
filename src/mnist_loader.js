"use strict";

let _ = require("lodash");
let linearAlgebra = require("linear-algebra")();
let Matrix = linearAlgebra.Matrix;

let lib = require("./lib");

class MnistLoader {

  static loadTrainingDataWrapper() {
    let trD = MnistLoader._loadTrainingData();
    let trainingInputs = trD[0].map(x => { return Matrix.reshape(x, 784, 1); });
    let trainingResults = trD[1].map(y => { return lib.vectorizedResult(y); });
    let trainingData = _.zip(trainingInputs, trainingResults);
    return trainingData;
  }

  static loadValidationDataWrapper() {
    let vaD = MnistLoader._loadValidationData();
    let validationInputs = vaD[0].map(x => { return Matrix.reshape(x, 784, 1); });
    let validationData = _.zip(validationInputs, vaD[1]);
    return validationData;
  }

  static loadTestDataWrapper() {
    let teD = MnistLoader._loadTestData();
    let testInputs = teD[0].map(x => { return Matrix.reshape(x, 784, 1); });
    let testData = _.zip(testInputs, teD[1]);
    return testData;
  }

  static _loadTrainingData() {
    let trainingInput = require("../data/training_input");
    let trainingOutput = require("../data/training_output");
    return [trainingInput, trainingOutput];
  }

  static _loadValidationData() {
    let validationInput = require("../data/validation_input");
    let validationOutput = require("../data/validation_output");
    return [validationInput, validationOutput];
  }

  static _loadTestData() {
    let testInput = require("../data/test_input");
    let testOutput = require("../data/test_output");
    return [testInput, testOutput];
  }

}

module.exports = MnistLoader;
