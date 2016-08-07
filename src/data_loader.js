'use strict';

const Promise = require('bluebird');
const fs = require('fs');
const zlib = require('zlib');
const path = require('path');

Promise.promisifyAll(fs);
Promise.promisifyAll(zlib);

class DataLoader {
  static loadTrainingData() {
    return Promise.all([
      DataLoader._loadDate('training_input.json.gz'),
      DataLoader._loadDate('training_output.json.gz')
    ]);
  }

  static loadValidationData() {
    return Promise.all([
      DataLoader._loadDate('validation_input.json.gz'),
      DataLoader._loadDate('validation_output.json.gz')
    ]);
  }

  static loadTestData() {
    return Promise.all([
      DataLoader._loadDate('test_input.json.gz'),
      DataLoader._loadDate('test_output.json.gz')
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

module.exports = DataLoader;
