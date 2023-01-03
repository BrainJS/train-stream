import { TrainStream, ITrainStreamNetwork } from './index';

import { NeuralNetwork, recurrent, utilities } from 'brain.js';
import { INumberHash } from 'brain.js/dist/src/lookup';
import { INeuralNetworkDatum } from 'brain.js/dist/src/neural-network';
import { INeuralNetworkState } from 'brain.js/dist/src/neural-network-types';

const { DataFormatter } = utilities;
const { LSTMTimeStep, LSTM } = recurrent;

describe('TrainStream', () => {
  const wiggle = 0.1;
  const errorThresh = 0.003;

  async function testTrainer<
    Network extends ITrainStreamNetwork<
      Parameters<Network['addFormat']>[0],
      Parameters<Network['trainPattern']>[0],
      Network['trainOpts']
      >
    >(
    neuralNetwork: Network,
    opts: {
      data: Array<Parameters<Network['addFormat']>[0]>;
      errorThresh?: number;
      iterations?: number;
    }
  ): Promise<INeuralNetworkState> {
    const { data } = opts;

    return await new Promise((doneTrainingCallback) => {
      /**
       * Every time you finish an epoch of flood call `trainStream.endInputs()`
       */
      function floodCallback(): void {
        for (let i = data.length - 1; i >= 0; i--) {
          trainStream.write(data[i]);
        }

        trainStream.endInputs();
      }

      const trainStream = new TrainStream({
        ...opts,
        neuralNetwork,
        floodCallback,
        doneTrainingCallback,
      });

      /**
       * kick off the stream
       */
      floodCallback();
    });
  }

  describe('using sparse training values', () => {
    it('can train fruit', async () => {
      const trainingData: Array<INeuralNetworkDatum<
        INumberHash,
        INumberHash
        >> = [
        { input: { apple: 1 }, output: { pome: 1 } },
        { input: { pear: 1 }, output: { pome: 1 } },
        { input: { hawthorn: 1 }, output: { pome: 1 } },
        { input: { peach: 1 }, output: { drupe: 1 } },
        { input: { plum: 1 }, output: { drupe: 1 } },
        { input: { cherry: 1 }, output: { drupe: 1 } },
        { input: { grape: 1 }, output: { berry: 1 } },
        { input: { tomato: 1 }, output: { berry: 1 } },
        { input: { eggplant: 1 }, output: { berry: 1 } },
        { input: { kiwis: 1 }, output: { berry: 1 } },
        { input: { persimmon: 1 }, output: { berry: 1 } },
        { input: { raspberry: 1 }, output: { aggregate: 1 } },
        { input: { blackberry: 1 }, output: { aggregate: 1 } },
        { input: { strawberry: 1 }, output: { aggregate: 1 } },
        { input: { watermelon: 1 }, output: { pepo: 1 } },
        { input: { cantaloupe: 1 }, output: { pepo: 1 } },
        { input: { cucumber: 1 }, output: { pepo: 1 } },
        { input: { squash: 1 }, output: { pepo: 1 } },
        { input: { lemon: 1 }, output: { modified: 1 } },
        { input: { orange: 1 }, output: { modified: 1 } },
      ];

      function largestKey(object: INumberHash) {
        let max = -Infinity;
        let maxKey = null;

        for (const key in object) {
          if (object[key] > max) {
            max = object[key];
            maxKey = key;
          }
        }

        return maxKey;
      }

      const neuralNetwork = new NeuralNetwork<INumberHash, INumberHash>();

      await testTrainer(neuralNetwork, { data: trainingData, errorThresh: 0.001 });
      for (const data of trainingData) {
        const output = neuralNetwork.run(data.input);
        const target = data.output;
        const outputKey = largestKey(output);
        const targetKey = largestKey(target);
        if (!outputKey || !targetKey) fail();
        expect(outputKey).toBe(targetKey);
        expect(
          output[outputKey] < target[targetKey] + wiggle &&
          output[outputKey] > target[targetKey] - wiggle
        ).toBeTruthy();
      }
    });
  });
  describe('bitwise functions', () => {
    describe('using arrays', () => {
      it('NOT function', async () => {
        const not = [
          {
            input: [0],
            output: [1],
          },
          {
            input: [1],
            output: [0],
          },
        ];
        const neuralNetwork = new NeuralNetwork<number[], number[]>();
        await testTrainer(neuralNetwork, { data: not, errorThresh });
        for (const i of not) {
          const output = neuralNetwork.run(i.input)[0];
          const target = i.output[0];

          expect(
            output < target + wiggle && output > target - wiggle
          ).toBeTruthy();
        }
      });

      it('XOR function', async () => {
        const xor = [
          {
            input: [0, 0],
            output: [0],
          },
          {
            input: [0, 1],
            output: [1],
          },
          {
            input: [1, 0],
            output: [1],
          },
          {
            input: [1, 1],
            output: [0],
          },
        ];
        const neuralNetwork = new NeuralNetwork<number[], number[]>();

        await testTrainer(neuralNetwork, { data: xor, errorThresh });
        for (const i of xor) {
          const output = neuralNetwork.run(i.input)[0];
          const target = i.output[0];

          expect(
            output < target + wiggle && output > target - wiggle
          ).toBeTruthy();
        }
      });

      it('OR function', async () => {
        const or = [
          {
            input: [0, 0],
            output: [0],
          },
          {
            input: [0, 1],
            output: [1],
          },
          {
            input: [1, 0],
            output: [1],
          },
          {
            input: [1, 1],
            output: [1],
          },
        ];
        const neuralNetwork = new NeuralNetwork<number[], number[]>();

        await testTrainer(neuralNetwork, { data: or, errorThresh });
        for (const i of or) {
          const output = neuralNetwork.run(i.input)[0];
          const target = i.output[0];

          expect(
            output < target + wiggle && output > target - wiggle
          ).toBeTruthy();
        }
      });

      it('AND function', async () => {
        const and = [
          {
            input: [0, 0],
            output: [0],
          },
          {
            input: [0, 1],
            output: [0],
          },
          {
            input: [1, 0],
            output: [0],
          },
          {
            input: [1, 1],
            output: [1],
          },
        ];
        const neuralNetwork = new NeuralNetwork<number[], number[]>();

        await testTrainer(neuralNetwork, { data: and, errorThresh });
        for (const i of and) {
          const output = neuralNetwork.run(i.input)[0];
          const target = i.output[0];

          expect(
            output < target + wiggle && output > target - wiggle
          ).toBeTruthy();
        }
      });
    });

    describe('objects', () => {
      it('AND function', async () => {
        interface MathInput extends INumberHash {
          left: number;
          right: number;
        }
        interface MathOutput extends INumberHash {
          product: number;
        }
        const and = [
          {
            input: { left: 0, right: 0 },
            output: { product: 0 },
          },
          {
            input: { left: 0, right: 1 },
            output: { product: 0 },
          },
          {
            input: { left: 1, right: 0 },
            output: { product: 0 },
          },
          {
            input: { left: 1, right: 1 },
            output: { product: 1 },
          },
        ];
        const neuralNetwork = new NeuralNetwork<MathInput, MathOutput>();

        await testTrainer(neuralNetwork, { data: and, errorThresh });
        for (const i of and) {
          const output = neuralNetwork.run(i.input).product;
          const target = i.output.product;

          expect(
            output < target + wiggle && output > target - wiggle
          ).toBeTruthy();
        }
      });
    });
  });

  describe('RNNTimeStep compatibility', () => {
    it('can average error for array,number, counting forwards and backwards', async () => {
      const iterations = 100;
      const data = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.8, 0.7, 0.6, 0.5, 0.4],
        [0.7, 0.6, 0.5, 0.4, 0.3],
        [0.6, 0.5, 0.4, 0.3, 0.2],
        [0.5, 0.4, 0.3, 0.2, 0.1],
      ];
      const neuralNetwork = new LSTMTimeStep({ hiddenLayers: [15] });

      const info = await testTrainer(neuralNetwork, { data, iterations });
      expect(info.error).toBeLessThan(0.05);
      expect(info.iterations).toBe(iterations);

      for (let i = 0; i < data.length; i++) {
        const value = data[i];
        expect(neuralNetwork.run(value.slice(0, 4)).toFixed(1)).toBe(
          value[4].toFixed(1)
        );
      }
    });
  });

  describe('RNN compatibility', () => {
    it('can average error for array,string, counting forwards and backwards', async () => {
      const iterations = 500;
      const data = [
        '12345',
        '23456',
        '34567',
        '45678',
        '56789',
        '98765',
        '87654',
        '76543',
        '65432',
        '54321',
      ].map(v => v.split(''));
      const neuralNetwork = new LSTM({
        hiddenLayers: [10],
        dataFormatter: new DataFormatter(data),
      });

      const info = await testTrainer(neuralNetwork, { data, iterations });
      expect(info.error).toBeLessThan(0.05);
      expect(info.iterations).toBe(iterations);

      for (let i = 0; i < data.length; i++) {
        const value = data[i];
        const output = neuralNetwork.run([value[0], value[1], value[2], value[3]]);
        expect(output).toBe(
          value[4]
        );
      }
    });
  });
});
