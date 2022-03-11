# brain.js train-stream
A node train stream for brain.js

## Usage

Streams are a very powerful tool in node for massive data spread across processes and are provided via the brain.js api in the following way:

```js
import { NeuralNetwork } from 'brain.js';
import { TrainStream } from 'train-stream';

const net = new NeuralNetwork();
const trainStream = new TrainStream({
  neuralNetwork: net,
  floodCallback: function () {
    flood(trainStream, data);
  },
  doneTrainingCallback: function (stats) {
    // network is done training!  What next?
  },
});

// kick it off
readInputs(trainStream, data);

function readInputs(stream, data) {
  for (let i = 0; i < data.length; i++) {
    stream.write(data[i]);
  }
  // let it know we've reached the end of the inputs
  stream.endInputs();
}
```

An example of using train stream can be found in [examples/javascript/stream-example.js](examples/javascript/stream-example.js)
