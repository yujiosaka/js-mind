
import assert from 'power-assert'

const linearAlgebra = require('linear-algebra')();
const { Matrix } = linearAlgebra;

const ___ = (...args) => new Matrix(args)

import {
  sigmoidPrime,
  vectorizedResult,
  dropoutLayer,
  norm,
  randn
} from '../src/lib'


describe('sigmoidPrime', ()=> {

  it('returns 0.25 when zero is given', ()=> {

    const zero = Matrix.zeros(1, 1)

    assert(zero.sigmoid().data[0][0] === 0.5)

    assert(sigmoidPrime(zero).data[0][0] === 0.5 * (-0.5 + 1))
  })

})

describe('vectorizedResult', ()=> {

  it('returns a 10-dimensional unit vector', ()=> {

    const result = vectorizedResult(3);

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].forEach(i => {
      const elem = result.data[i][0]
      const expected = (i === 3) ? 1 : 0
      assert(elem === expected)
    })
  })
})

describe('dropoutLayer', ()=> {

    const mat = ___(
      [1,4,9],
      [3,8,2],
      [6,5,7],
    )

  it('always drops out values from a layer when pDropout is 1', ()=> {

    const pDropout = 1 // probability to dropout
    const result = dropoutLayer(mat, pDropout)

    assert.deepEqual(result, Matrix.zeros(3, 3))
  })

  it('never drops out values from a layer when pDropout is 0', ()=> {

    const pDropout = 0
    const result = dropoutLayer(mat, pDropout)

    assert.deepEqual(result, mat)
  })
})


describe('norm', ()=> {

  it('always returns the mean when sigma is 0', ()=> {

    assert(norm(123, 0) === 123)
  })

  it('returns different values every time', ()=> {
    assert(norm(123, 1) !== norm(123, 1))
  })


})


describe('randn', ()=> {

  it('returns matrix, all of whose elements follow the standard normal distribution', ()=> {

    const result = randn(3, 3)

    const Z = 4.0 // p < 1.0e-4

    result.map(elem => {
      assert(Math.abs(elem) < Z)
    })

  })
})
