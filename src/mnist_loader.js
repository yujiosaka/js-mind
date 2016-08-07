'use strict';

const _ = require('lodash');
const { Matrix } = require('linear-algebra')();

const DataLoader = require('./data_loader')

class MnistLoader {
  static loadTrainingDataWrapper() {
    return DataLoader.loadTrainingData().then(trD => {
      const trainingInputs = trD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      const trainingResults = trD[1].map(y => { return MnistLoader._vectorizedResult(y); });
      const trainingData = _.zip(trainingInputs, trainingResults);
      return trainingData;
    });
  }

  static loadValidationDataWrapper() {
    return DataLoader.loadValidationData().then(vaD => {
      const validationInputs = vaD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      const validationData = _.zip(validationInputs, vaD[1]);
      return validationData;
    });
  }

  static loadTestDataWrapper() {
    return DataLoader.loadTestData().then(teD => {
      const testInputs = teD[0].map(x => { return Matrix.reshape(x, 784, 1); });
      const testData = _.zip(testInputs, teD[1]);
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
