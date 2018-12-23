const tfn = require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

// Step 1: Set up DATA
const Xvalues = []
const YValues = []
for (let index = 1; index < 20; index++) {
  Xvalues.push(index)
  YValues.push(index * 3)
}

const xs = tf.tensor(Xvalues);
const ys = tf.tensor(YValues);

// Step 2: create model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Step 3: set the optimizer
const learningRate = 0.005;
const optimizer = tf.train.sgd(learningRate);

// Step 4: fit the model with the datas
model.compile({loss: 'meanSquaredError',optimizer: optimizer});
model.fit(xs,ys ,{epochs: 500}).then(() => {
  model.predict(tf.tensor2d([12],[1,1])).print();
  ys.print()
});
