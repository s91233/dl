const tf = require('@tensorflow/tfjs'); require('@tensorflow/tfjs-node');
const fs = require('fs');
const fetch = require('node-fetch');
const { Tokenizer } = require('./tokenizer');
const tokenizer = new Tokenizer();

let VOCAB_SIZE;
const SEQ_LENGTH = 10;
const EMBEDDING_SIZE = 50;
const LSTM_UNITS = 100;
const BATCH_SIZE = 32;
const EPOCHS = 100;

// Load your data from a file
async function loadDataFile() {
    return fs.readFileSync('pg1513.txt','utf8');
}

// Load your data
async function loadData() {
    const response = await fetch('https://www.gutenberg.org/ebooks/1513.txt.utf-8');
    return await response.text();
}

function preprocessData(text) {
    // Fit the tokenizer on the cleaned text
    tokenizer.fitOnTexts([text]);
    // Tokenize your text and create sequences
    const sequences = tokenizer.textsToSequences([text]);
    // Update VOCAB_SIZE to the actual vocabulary size
    VOCAB_SIZE = Object.keys(tokenizer.wordIndex).length;

    return { sequences: sequences, wordIndex: tokenizer.wordIndex };
}

// Build your model
function buildModel(vocabSize, embeddingSize, lstmUnits) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({inputDim: vocabSize, outputDim: embeddingSize}));
    model.add(tf.layers.lstm({units: lstmUnits, returnSequences: true}));
    model.add(tf.layers.lstm({units: lstmUnits}));
    model.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
    return model;
}

// Train your model
async function trainModel(model, sequences, batchSize, epochs) {
    // Prepare your sequences for training
    // FIXME
    model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
    return model.fit(xs, ys, {batchSize, epochs});
}

// Generate text
function generateText(model, wordIndex, seed, length) {
    // ToDo
    return "";
}

// Main function
async function main() {
    const text = await loadData();
    const { sequences, wordIndex } = preprocessData(text);
    const model = buildModel(VOCAB_SIZE, EMBEDDING_SIZE, LSTM_UNITS);
    const history = await trainModel(model, sequences, BATCH_SIZE, EPOCHS);
    await model.save('file://./model');
    const loadedModel = await tf.loadLayersModel('file://./model/model.json');
    const seed = "Enter your seed text here";
    console.log(generateText(loadedModel, wordIndex, seed, SEQ_LENGTH));
}

main();