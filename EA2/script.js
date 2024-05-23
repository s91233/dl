function generateData(N) {
    return Array.from({ length: N }, () => {
      const x = Math.random() * 4 - 2; // [-2,+2]
      return { x, y: 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1 };
    });
}

function gaussianNoise(data, variance) {
    return data.map(d => ({ x: d.x, y: d.y + Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) * Math.sqrt(variance) }));
}

function splitData(data) {
    return { trainData: data.slice(0, data.length / 2), testData: data.slice(data.length / 2) };
}

function createModel(layers, units, activation) {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: 1}));
    Array.from({ length: layers }, () => model.add(tf.layers.dense({ units, activation })));
    model.add(tf.layers.dense({units: 1}));
    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        const inputs = data.map(d => d.x);
        const labels = data.map(d => d.y);
        return { 
            inputs: tf.tensor2d(inputs, [inputs.length, 1]),
            labels: tf.tensor2d(labels, [labels.length, 1]) 
        };
    });
}

async function trainModel(model, tensors, config) {
    model.compile({
        metrics: ['mse'],
        optimizer: config.optimizer,
        loss: tf.losses.meanSquaredError,
    });
    model.optimizer.learningRate = config.rate;
    await model.fit(tensors.inputs, tensors.labels, {
        batchSize: config.batch,
        epochs: config.epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training' },
            ['loss', 'mse'],
            { callbacks: ['onEpochEnd'] }
        ),
    });
}

function predictValues(model, data) {
    return tf.tidy(() => {
        const xs = data.map(d => d.x);
        return data.map((d,i) => ({
            x: d.x, y: (model.predict(tf.tensor2d(xs, [xs.length, 1])).dataSync())[i]
        }));
    });
}

function plotData(id, values, series, xLabel, yLabel) {
    tfvis.render.scatterplot(document.getElementById(id), { values, series }, { xLabel, yLabel });
}

function testModel(model, data, id) {
    const { inputs, labels } = convertToTensor(data);
    const mse = (tf.losses.meanSquaredError(labels, model.predict(inputs))).mean().arraySync();
    plotData(id, [data, predictValues(model, data)], ['Original', 'Predicted'], `MSE: ${mse.toFixed(4)}`, '');
}

async function downloadModel(modelName) {
    const model = await tf.loadLayersModel(`localstorage://${modelName}`);
    await model.save(`downloads://${modelName}`);
}

async function uploadModel() {
    const files = document.getElementById('upload').files;
    if (!files.length) return;
    const model = await tf.loadLayersModel(tf.io.browserFiles([files[0]]));

    const form = document.getElementById('config');
    const originalData = generateData(parseInt(form.N.value));
    const noisyData = gaussianNoise(originalData, parseFloat(form.variance.value));

    testModel(model, originalData, 'uploaded-original');
    testModel(model, noisyData, 'uploaded-noisy');
}

async function run()
{
    const form = document.getElementById('config');

    const [rate, variance] = ['rate', 'variance'].map(field => parseFloat(form[field].value));
    const [activation, optimizer] = ['activation', 'optimizer'].map(field => form[field].value);
    const [N, layers, neurons, batch, epochs, overfit] = ['N', 'layers', 'neurons', 'batch', 'epochs', 'overfit'].map(field => parseInt(form[field].value));
    
    const originalData = generateData(N);
    const { trainData: trainOriginal, testData: testOriginal } = splitData(originalData);
    plotData('data-original', [trainOriginal, testOriginal], ['Train', 'Test'], 'x', 'y');

    const noisyData = gaussianNoise(originalData, variance);
    const { trainData: trainNoisy, testData: testNoisy } = splitData(noisyData);
    plotData('data-noisy', [trainNoisy, testNoisy], ['Train', 'Test'], 'x', 'y');

    const modelOriginal = createModel(layers, neurons, activation);
    tfvis.show.modelSummary({name: 'Layers'}, modelOriginal);

    const config = { epochs, batch, rate, optimizer };

    await trainModel(modelOriginal, convertToTensor(trainOriginal), config);
    modelOriginal.save('localstorage://model-original');
    testModel(modelOriginal, trainOriginal, 'original-train');
    testModel(modelOriginal, testOriginal, 'original-test');

    const modelNoisyBest = createModel(layers, neurons, activation);
    await trainModel(modelNoisyBest, convertToTensor(trainNoisy), config);
    modelNoisyBest.save('localstorage://model-noisy-best');
    testModel(modelNoisyBest, trainNoisy, 'best-train');
    testModel(modelNoisyBest, testNoisy, 'best-test');

    config.epochs *= overfit;

    const modelNoisyOverfit = createModel(layers, neurons, activation);
    await trainModel(modelNoisyOverfit, convertToTensor(trainNoisy), config);
    modelNoisyOverfit.save('localstorage://model-noisy-overfit');
    testModel(modelNoisyOverfit, trainNoisy, 'overfit-train');
    testModel(modelNoisyOverfit, testNoisy, 'overfit-test');
}

document.addEventListener('DOMContentLoaded', run);