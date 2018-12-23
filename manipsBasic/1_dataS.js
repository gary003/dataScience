const tf = require('@tensorflow/tfjs')
const tfn = require('@tensorflow/tfjs-node')

//step 0 : data
const Xvalues = []
const YValues = []
for (let index = 1; index < 20; index++) {
  Xvalues.push(index)
  YValues.push(index * 4)
}

const xs = tf.tensor(Xvalues);
const ys = tf.tensor(YValues);

// Step 1: Set up Variables
const a = tf.variable(tf.tensor([0]))
const d = tf.variable(tf.tensor([5]))

// Step 2: Build a prediction  Model : y = a * x + d
const predict = (x) => {
  return tf.tidy(() => {
    return a.mul(x).add(d);
  });
}

// Step 3: Train the Model
const loss = (predictions, labels) => {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  return predictions.sub(labels).square().mean();
}

const train = async (xs_p, ys_p, numIterations = 1565) => {
  const learningRate = 0.007;
  const optimizer = tf.train.sgd(learningRate);
  
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const lossT = loss(predict(xs_p), ys_p)
      lossT.print()
      return lossT
    })
    //await tf.nextFrame();
  }
}

train(xs,ys).then(() => {
  a.print()
  d.print()
  predict(tf.tensor1d([16])).print()
  console.log(tf.memory().numTensors)
})
