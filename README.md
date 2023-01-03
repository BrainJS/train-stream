# brain.js train-stream
A node train stream for brain.js

## Usage

Streams are a very powerful tool in node for massive data spread across processes and are provided via the brain.js api in the following way:

### Use with brain.js' NeuralNetwork class
```typescript
import { NeuralNetwork } from 'brain.js';
import { TrainStream } from 'train-stream';

const net = new NeuralNetwork();
const trainStream = new TrainStream({
  neuralNetwork: net,
  floodCallback: () => {
    readInputs(trainStream, data);
  },
  doneTrainingCallback: (stats) => {
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

### Use with brain.js' RNN/LSTM class
```typescript
import { recurrent, utilities } from 'brain.js';
import { TrainStream } from 'train-stream';
const { LSTM } = recurrent;

const neuralNetwork = new LSTM({
  hiddenLayers: [10],
  dataFormatter: new utilities.DataFormatter(), // You'll need to setup a dataformatter
});
const trainStream = new TrainStream({
  neuralNetwork,
  floodCallback: () => {
    trainStream.write(myData);
    trainStream.endInputs();
  },
  doneTrainingCallback: (stats) => {
    // network is done training!  What next?
  },
});

```

An example of using train stream can be found in [examples/stream-example.ts](examples/stream-example.ts)
