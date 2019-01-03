const tfn = require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

// Step 1: Set up DATA
/* !! Data must be randomize !!
 the learn must not be affected by the order
*/
// const Yvalues = [1,2,3,4,5, 6, 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
// const Xvalues = [2,3,5,7,11,13,17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89 ,97]
const Yvalues = []
const Xvalues = [] 
for (let index = 1; index < 20; index++) {
  const num = Math.floor(Math.random() * 10) + 1
  Xvalues.push(num)
  Yvalues.push(num * 3)
}

const xs = tf.tensor(Xvalues);
const ys = tf.tensor(Yvalues);

// Step 2: create model
const model = tf.sequential();
const hiddenLayer = tf.layers.dense({units: 1, inputShape: [1]})
model.add(hiddenLayer);

// Step 3: set the optimizer
const learningRate = 0.02 ;
const optimizer = tf.train.sgd(learningRate);

// Step 4: fit the model with the datas
model.compile({loss: 'meanSquaredError',optimizer: optimizer});
model.fit(xs,ys ,{epochs: 13000}).then(() => {
  model.predict(tf.tensor2d([12],[1,1])).print();
  ys.print()
});
