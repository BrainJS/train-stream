import commonjs from '@rollup/plugin-commonjs';
import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import * as pkg from './package.json';

const extensions = ['.mjs', '.js', '.json', '.node', '.ts'];

export default {
  input: './src/index.ts',

  plugins: [
    // Allows node_modules resolution
    resolve({
      preferBuiltins: true,
      browser: false,
      extensions,
    }),

    // allow json importing
    json(),

    // Allow bundling cjs modules. Rollup doesn't understand cjs
    commonjs(),

    // Compile TypeScript/JavaScript files
    typescript(),
  ],

  output: [
    {
      file: pkg.main,
      format: 'cjs',
      sourcemap: true,
    },
    {
      file: pkg.module,
      format: 'es',
      sourcemap: true,
    },
  ],
};
