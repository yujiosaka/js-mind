'use strict';

let _ = require('lodash');
let linearAlgebra = require('linear-algebra')();
let Matrix = linearAlgebra.Matrix;

let DataLoader = require('./data_loader')

class MnistLoader {
  static loadTrainingDataWrapper() {
    return DataLoader.loadTrainingData().then(trD => {
      let trainingInputs = trD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let trainingResults = trD[1].map(y => { return MnistLoader._vectorizedResult(y); });
      let trainingData = _.zip(trainingInputs, trainingResults);
      return trainingData;
    });
  }

  static loadValidationDataWrapper() {
    return DataLoader.loadValidationData().then(vaD => {
      let validationInputs = vaD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let validationData = _.zip(validationInputs, vaD[1]);
      return validationData;
    });
  }

  static loadTestDataWrapper() {
    return DataLoader.loadTestData().then(teD => {
      let testInputs = teD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      let testData = _.zip(testInputs, teD[1]);
      return testData;
    });
  }

  static _vectorizedResult(j) {
    const e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => [0]);
    e[j] = [1];
    return new Matrix(e);
  }
}

module.exports = MnistLoader;
