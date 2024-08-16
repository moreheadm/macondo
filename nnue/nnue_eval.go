package nnue

import (
	"gonum.org/v1/gonum/mat"
)


type struct LinearLayer {
	weights mat.Dense
	bias mat.VecDense
}




type NnueEvaluator struct {
	layers []LinearLayer
	activations []func(vec * mat.VecDense)
}




func (vec *mat.VecDense) Clamp(minValue, maxValue float64) {
	for i := 0, i < vec.Len(); i++ {
		if vec.AtVec(i) < minValue {
			vec.SetVec(i, minValue)
		} else if inputVector.AtVec(i) > maxValue {
			vec.SetVec(i, maxValue)
		}
	}
}

func (vec *mat.VecDense) SparseVectorMatMul(inputVector mat.VecDense, matrix mat.Dense) {
	_, cols := matrix.Dims()
	rowVector := mat.NewVecDense(cols, nil)

	// Iterate over each element in the input vector
	for i := 0; i < inputVector.Len(); i++ {
		inputValue := inputVector.AtVec(i)
		if inputValue == 0 {
			continue // Skip if inputValue is zero
		}

		// Get the i-th row from the matrix
		matrix.Row(rowVector.RawVector().Data, i)

		// Add the row vector to the output vector
		vec.AddScaleVec(vec, inputValue, rowVector)
	}
}

func ClampedRelu(vec * mat.VecDense) {
	vec.Clamp(0, 1)
}

func Identity(vec * mat.VecDense) {}

func (ne * NnueEvaluator) Eval(inputVector mat.VecDense) float64 {
	var currVector := &inputVector
	for i, layer := range ne.layers {
		outputVector := NewVecDense(layer.OutputDim(), nil)
		outputVector.SparseVectorMatMul(inputVector)
	}
}
