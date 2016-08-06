"use strict";

let Promise = require("bluebird");
let fs = require("fs");
let zlib = require("zlib");
let path = require("path");
Promise.promisifyAll(fs);
Promise.promisifyAll(zlib);

let _ = require("lodash");
let linearAlgebra = require("linear-algebra")();
let Matrix = linearAlgebra.Matrix;

let lib = require("./lib");

class MnistLoader {
  static loadTrainingDataWrapper() {
    return MnistLoader._loadTrainingData().then(trD => {
      let trainingInputs = trD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let trainingResults = trD[1].map(y => { return lib.vectorizedResult(y); });
      let trainingData = _.zip(trainingInputs, trainingResults);
      return trainingData;
    });
  }

  static loadValidationDataWrapper() {
    return MnistLoader._loadValidationData().then(vaD => {
      let validationInputs = vaD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let validationData = _.zip(validationInputs, vaD[1]);
      return validationData;
    });
  }

  static loadTestDataWrapper() {
    return MnistLoader._loadTestData().then(teD => {
      let testInputs = teD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let testData = _.zip(testInputs, teD[1]);
      return testData;
    });
  }

  static _loadTrainingData() {
    return Promise.all([
      MnistLoader._loadDate('training_input.json.gz'),
      MnistLoader._loadDate('training_output.json.gz')
    ]);
  }

  static _loadValidationData() {
    return Promise.all([
      MnistLoader._loadDate('validation_input.json.gz'),
      MnistLoader._loadDate('validation_output.json.gz')
    ]);
  }

  static _loadTestData() {
    return Promise.all([
      MnistLoader._loadDate('test_input.json.gz'),
      MnistLoader._loadDate('test_output.json.gz')
    ]);
  }

  static _loadDate(filename) {
    return fs.readFileAsync(
      path.join(__dirname, `../data/${filename}`)
    ).then(content => {
      return zlib.gunzipAsync(content);
    }).then(binary => {
      return JSON.parse(binary.toString());
    });
  }
}

module.exports = MnistLoader;
