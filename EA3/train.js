const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');
const { Tokenizer } = require('tf_node_tokenizer');
const { pad_sequences } = require('./util');
const csvWriter = require('csv-writer').createObjectCsvWriter;
const { ModelCheckpoint, CsvLogger } = require('tfjs_ports')

let model;

function createDirectory(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function saveModel(model, path) {
    return model.save(`file://${path}`);
}

function loadModel(path) {
    return tf.loadLayersModel(`file://${path}/model.json`);
}

function writeToFile(path, data) {
    fs.writeFileSync(path, JSON.stringify(data));
}

/**
 * @param {number} dict_size
 * @param {number} units
 * @param {number} lr
 */
function buildModel(dict_size, units = 200, lr = 0.01) {
    let model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: dict_size, outputDim: units }));
    model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units }) }));
    model.add(tf.layers.dense({ units: dict_size, activation: 'softmax' }));
    model.compile({ loss: 'categoricalCrossentropy', metrics: ['categoricalAccuracy'], optimizer: tf.train.adam(lr) });
    model.summary();
    return model;
}

async function trainModel() {
    // Tokenize
    const text = fs.readFileSync('./data/pg1513.txt', 'utf-8');
    const DIR = path.basename('./data/pg1513.txt').split('.')[0] + '_' + new Date().toISOString().replace(/:/g, '-');
    createDirectory('./models/' + DIR);

    const tokenizer = new Tokenizer();
    tokenizer.fitOnTexts([text]);
    const dict_size = Object.keys(tokenizer.word_index).length + 1;

    let sequences = text.split('\n').flatMap(line => {
        const sequence = tokenizer.textsToSequences([line])[0];
        return Array.from({length: sequence.length}, (_, i) => sequence.slice(0, i + 1));
    });

    const max_sequence_len = Math.max(...sequences.map(seq => seq.length));
    sequences = pad_sequences(sequences);

    const X = tf.tensor(sequences.map(seq => seq.slice(0, -1)), null, 'int32');
    const Y = tf.tensor(sequences.map(seq => seq.slice(-1)), null, 'int32');

    writeToFile(`./models/${DIR}/dict.json`, {
        word_index: tokenizer.word_index,
        index_word: tokenizer.index_word,
    });

    const writer = csvWriter({
        path: `./models/${DIR}/logs.csv`,
        header: [
            { id: 'epoch', title: 'Epoch' },
            { id: 'loss', title: 'Loss' },
            { id: 'acc', title: 'Accuracy' },
            { id: 'test', title: 'Test' }
        ],
        append: true // Append to existing file
    });
    model = buildModel(dict_size);

    let best = { acc: 0, epoch: 0 };
    const history = await model.fit(X, tf.oneHot(Y.squeeze(), dict_size), {
        epochs: process.argv.slice(2)[0] ?? 100, batchSize: 64, callbacks: {
            onEpochEnd: async function (epoch, logs) {
                let test = 'O serpent heart';
                for (let i = 0; i < 10; i++) {
                    const sequence = pad_sequences([tokenizer.textsToSequences([test])[0]], max_sequence_len - 1)[0];
                    test += " " + (tokenizer.index_word[tf.argMax(model.predict(tf.tensor([sequence])).dataSync()).dataSync()[0]] || "");
                }
                writer.writeRecords([{epoch, loss: logs.loss, acc: logs.categoricalAccuracy, test}])
                    .then(() => {
                        console.log(epoch,logs.loss,logs.categoricalAccuracy,test)
                    })
                    .catch((error) => {
                        console.error('Error writing to CSV file:', error);
                    });
                if (logs.categoricalAccuracy > best.acc) {
                    best.acc = logs.categoricalAccuracy;
                    best.epoch = epoch;
                    await saveModel(model, `./models/${DIR}/${epoch}`);
                }}, onTrainEnd: async () =>
                model = await loadModel(`./models/${DIR}/${best.epoch}`)
        }
    });
    writeToFile(`./models/${DIR}/history.json`, history);
    // You can add more lines of code here that will execute in sequence
    await saveModel(model, `./models/${DIR}/tfjs`);
    // Test
}

trainModel().catch(console.error);
