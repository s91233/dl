const { pad_sequences, plot_history } = require('./util');
const { tokenizerFromJson } = require('tf_node_tokenizer');
// ... (other code)

window.tfmodel = null;

let inputs = [];
let models = [];

const n = (window.innerWidth || document.documentElement.clientWidth) < 768 ? 8 : 16;

document.addEventListener('DOMContentLoaded', async () => {
    inputs = await (await fetch('data.json')).json();
    populateSelect(document.getElementById('input-select'), inputs);
    await loadText(0);

    document.getElementById('input-select').addEventListener('change', (event) => {
        loadText(event.target.value);
    });

    document.getElementById('model-select').addEventListener('change', (event) => {
        loadModel(event.target.value);
    });

    document.getElementById('user-input').addEventListener("input", resizeTextbox, false);
    document.getElementById('predict-btn').addEventListener('click', handlePrediction);
    document.getElementById('continue-btn').addEventListener('click', predictWord);
    document.getElementById('auto-btn').addEventListener('click', predictWords);
    document.getElementById('stop-btn').addEventListener('click', stop);
    document.getElementById('reset-btn').addEventListener('click', reset);

    document.getElementById("text-input-form").addEventListener("keyup", event => {
        if(event.key !== "Enter") return;
        event.preventDefault();
        predictWord()
    });
});

function populateSelect(select, items) {
    select.innerHTML = '';
    items.forEach((item, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = item.name;
        select.appendChild(option);
    });
}

async function loadText(index) {
    reset();
    const selectedText = inputs[index];
    if (selectedText.models && selectedText.models.length > 0) {
        models = selectedText.models;
        populateSelect(document.getElementById('model-select'), selectedText.models);
        await loadModel(0);
    }

    document.getElementById('examples').innerHTML = `<ul>${(selectedText.examples || []).map((example, index) =>
        `<li><a href="#" class="example">${example}</a></li>`).join('')}</ul>`;
    document.querySelectorAll('.example').forEach(button => {
        button.addEventListener('click', (event) => {
            document.getElementById('user-input').value = event.target.textContent; predictWords();
        });
    });
}

async function loadModel(index) {
    reset();
    document.body.style.cursor = 'wait';
    const selectedModel = models[index];
    document.getElementById('max-sequence').value = selectedModel.maxSequenceLength ?? -1;

    const model = await tf.loadLayersModel(`${selectedModel.path}/model.json`);
    console.log('Model loaded:', model);
    window.tfmodel = model;

    await (window.tokenizer = tokenizerFromJson(await (await fetch(`${selectedModel.path}/dict.json`)).text()));

    document.getElementById('predictions-table').innerHTML = null;
    const modelInfo = document.getElementById('model-info');
    modelInfo.innerHTML = model.layers.map(layer => `<span>${layer.name}: ${layer.outputShape}</span>`).join('<br>');

    const canvas = document.getElementById('history').getContext('2d');
    await plot_history(canvas, selectedModel.path);

    await model.predict(tf.zeros([1, parseInt(document.getElementById('max-sequence').value, 10)])).dataSync();
    document.body.style.cursor = 'auto';
}

async function handlePrediction(event) {
    event.preventDefault();
    const text = document.getElementById('user-input').value;
    const numPreds = parseInt(document.getElementById('num-preds').value, 10);
    const predictions = await generatePredictions(text, numPreds);
    document.getElementById('continue-btn').disabled = false;
    updatePredictionsTable(predictions);
}

async function generatePredictions(text, numPredictions) {
    const maxSeqLen = parseInt(document.getElementById('max-sequence').value, 10);
    const sequence = pad_sequences([window.tokenizer.textsToSequences([text])[0]], maxSeqLen)[0];
    const predictions = window.tfmodel.predict(tf.tensor([sequence])).dataSync();
    calculatePerplexity(predictions)

    return Array.from(predictions)
        .map((probability, index) => ({word: window.tokenizer.index_word[index], probability}))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, numPredictions);
}

function updatePredictionsTable(predictions) {
    document.getElementById('predictions-table').innerHTML = predictions.map(prediction =>
        `<tr><td><a href="#" class="prediction">${prediction.word}</a></td><td>${prediction.probability.toFixed(n)}</td></tr>`
    ).join('');
    document.querySelectorAll('.prediction').forEach(button => {
        button.addEventListener('click', () => selectWord(button.textContent));
    });
}

function resizeTextbox() {
    document.getElementById('user-input').style.height = 'auto';
    document.getElementById('user-input').style.height = (this.scrollHeight) + "px";
}

async function selectWord(word) {
    const userInput = document.getElementById('user-input');
    userInput.value = userInput.value ? userInput.value + ' ' + word : word;
    userInput.dispatchEvent(new Event('input', { bubbles: true }));
    await handlePrediction(new Event('click'));
}

async function predictWord() {
    const text = document.getElementById('user-input').value;
    const predictions = await generatePredictions(text, 1);
    if (predictions.length > 0) await selectWord(predictions[0].word)
    await new Promise(resolve => setTimeout(resolve, 1));
}

let auto = false;
async function predictWords() {
    auto = true;
    document.getElementById('auto-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';
    const numWords = parseInt(document.getElementById('num-words').value, 10);
    for (let i = 0; i < numWords && auto; i++) await predictWord(); auto = false;
    document.getElementById('auto-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
}

function stop() {
    auto = false;
}

function reset() {
    document.getElementById('predictions-table').innerHTML = '';
    document.getElementById('continue-btn').disabled = true;
    document.getElementById('user-input').value = '';
    document.getElementById('num-words').value = 10
    document.getElementById('num-preds').value = 5;
    document.getElementById('perplexity').innerText = parseInt(0).toFixed(n);
    perplexity = { log: 0, words: 0, result: 0 };
    resizeTextbox()
}

let perplexity = { log: 0, words: 0, result: 0 };
function calculatePerplexity(predictions) {
    perplexity.log += Math.log(predictions[tf.argMax(predictions).dataSync()[0]]);
    perplexity.words++;
    perplexity.result = Math.exp(-perplexity.log / perplexity.words);
    document.getElementById('perplexity').innerText = perplexity.result.toFixed(n);
}
