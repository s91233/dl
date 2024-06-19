const { Tokenizer, tokenizerFromJson } = require('./tokenizer');
// Example usage
const tokenizer = new Tokenizer();
// ... (other code)
window.Tokenizer = Tokenizer;

let VOCAB_SIZE;
const SEQ_LENGTH = 10;
const EMBEDDING_SIZE = 50;
const LSTM_UNITS = 100;
const BATCH_SIZE = 32;
const EPOCHS = 10;

// Load your data
async function loadData() {
    const response = await fetch('pg1513.txt');
    return await response.text();
}

// Preprocess your data
function preprocessData(text) {
    // Remove all characters except for a-z, A-Z, spaces and umlauts
    const cleanedText = text.replace(/[^a-zA-ZäöüÄÖÜß ]/g, ' ').replace(/\s+/g, ' ');

    // Fit the tokenizer on the cleaned text
    tokenizer.fitOnTexts([cleanedText]);
    // Tokenize your text and create sequences
    const sequences = tokenizer.textsToSequences([cleanedText]);
    // Update VOCAB_SIZE to the actual vocabulary size
    VOCAB_SIZE = Object.keys(tokenizer.wordIndex).length;

    return {sequences, wordIndex: tokenizer.wordIndex};
}

function _preprocessData(text) {
    // Remove all characters except for a-z, A-Z, spaces and umlauts
    const cleanedText = text.replace(/[^a-zA-ZäöüÄÖÜß ]/g, ' ').replace(/\s+/g, ' ');

    // Tokenize your text and create sequences
    const words = cleanedText.split(' ');
    const wordIndex = {};
    const sequences = [];
    let wordCount = 0;
    words.forEach(word => {
        if (!(word in wordIndex)) {
            wordIndex[word] = wordCount++;
        }
        sequences.push(wordIndex[word]);
    });

    // Update VOCAB_SIZE to the actual vocabulary size
    VOCAB_SIZE = Object.keys(wordIndex).length;

    return {sequences, wordIndex};
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
    const xs = tf.tensor2d(sequences.slice(0,-1), [sequences.length - 1, 1]);
    const ys = tf.oneHot(sequences.slice(1), VOCAB_SIZE);
    const fitCallbacks = tfvis.show.fitCallbacks({name: 'Training'}, ['loss','accuracy','mse'], {callbacks: ['onEpochEnd']});
    model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
    return await model.fit(xs, ys, {batchSize, epochs, verbose: 0});
}

// Generate text
function generateText(model, wordIndex, seed, length) {
    let sequence = seed.split(' ').map(word => wordIndex[word]);
    for (let i = 0; i < length; i++) {
        const prediction = model.predict(tf.tensor2d(sequence, [1, sequence.length]));
        sequence.push(prediction.argMax(-1).dataSync()[0]);
    }
    return sequence.map(index => Object.keys(wordIndex).find(word => wordIndex[word] === index)).join(' ');
}

document.getElementById('train').addEventListener('click', async () => {
    const text = await loadData();
    const { sequences } = preprocessData(text);
    const model = buildModel(VOCAB_SIZE, EMBEDDING_SIZE, LSTM_UNITS);
    tfvis.show.modelSummary({name: 'Layers'}, model);
    await trainModel(model, sequences, BATCH_SIZE, EPOCHS);
    model.save('localstorage://model');
});

document.getElementById('test').addEventListener('click', async () => {
    const model = await tf.loadLayersModel('localstorage://model').catch(() => null);
    if (!model) return document.getElementById('output').innerText = 'Error: No model trained.';
    const wordIndex = preprocessData(await loadData()).wordIndex;
    const seed = document.getElementById('input').value;
    document.getElementById('output').innerHTML = generateText(model, wordIndex, seed, SEQ_LENGTH);
});