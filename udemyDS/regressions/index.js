const tf = require('@tensorflow/tfjs')
const tfn = require('@tensorflow/tfjs-node')
const loadCSV = require('./load-csv')

let {features , labels , testFeatures , testLabels} = loadCSV('./cars.csv',{
  shuffle : true ,
  splitTest : 50,
  dataColumns : ['horsepower'],
  labelColumns : ['mpg']
})

console.log(testLabels.length);
