/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "./script.js":
/*!*******************!*\
  !*** ./script.js ***!
  \*******************/
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {

eval("const { Tokenizer, tokenizerFromJson } = __webpack_require__(/*! ./tokenizer */ \"./tokenizer.js\");\n// Example usage\nconst tokenizer = new Tokenizer();\n// ... (other code)\nwindow.Tokenizer = Tokenizer;\n\nlet VOCAB_SIZE;\nconst SEQ_LENGTH = 10;\nconst EMBEDDING_SIZE = 50;\nconst LSTM_UNITS = 100;\nconst BATCH_SIZE = 32;\nconst EPOCHS = 10;\n\n// Load your data\nasync function loadData() {\n    const response = await fetch('pg1513.txt');\n    return await response.text();\n}\n\n// Preprocess your data\nfunction preprocessData(text) {\n    // Remove all characters except for a-z, A-Z, spaces and umlauts\n    const cleanedText = text.replace(/[^a-zA-ZäöüÄÖÜß ]/g, ' ').replace(/\\s+/g, ' ');\n\n    // Fit the tokenizer on the cleaned text\n    tokenizer.fitOnTexts([cleanedText]);\n    // Tokenize your text and create sequences\n    const sequences = tokenizer.textsToSequences([cleanedText]);\n    // Update VOCAB_SIZE to the actual vocabulary size\n    VOCAB_SIZE = Object.keys(tokenizer.wordIndex).length;\n\n    return {sequences, wordIndex: tokenizer.wordIndex};\n}\n\nfunction _preprocessData(text) {\n    // Remove all characters except for a-z, A-Z, spaces and umlauts\n    const cleanedText = text.replace(/[^a-zA-ZäöüÄÖÜß ]/g, ' ').replace(/\\s+/g, ' ');\n\n    // Tokenize your text and create sequences\n    const words = cleanedText.split(' ');\n    const wordIndex = {};\n    const sequences = [];\n    let wordCount = 0;\n    words.forEach(word => {\n        if (!(word in wordIndex)) {\n            wordIndex[word] = wordCount++;\n        }\n        sequences.push(wordIndex[word]);\n    });\n\n    // Update VOCAB_SIZE to the actual vocabulary size\n    VOCAB_SIZE = Object.keys(wordIndex).length;\n\n    return {sequences, wordIndex};\n}\n\n// Build your model\nfunction buildModel(vocabSize, embeddingSize, lstmUnits) {\n    const model = tf.sequential();\n    model.add(tf.layers.embedding({inputDim: vocabSize, outputDim: embeddingSize}));\n    model.add(tf.layers.lstm({units: lstmUnits, returnSequences: true}));\n    model.add(tf.layers.lstm({units: lstmUnits}));\n    model.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));\n    return model;\n}\n\n// Train your model\nasync function trainModel(model, sequences, batchSize, epochs) {\n    // Prepare your sequences for training\n    const xs = tf.tensor2d(sequences.slice(0,-1), [sequences.length - 1, 1]);\n    const ys = tf.oneHot(sequences.slice(1), VOCAB_SIZE);\n    const fitCallbacks = tfvis.show.fitCallbacks({name: 'Training'}, ['loss','mse'], {callbacks: ['onEpochEnd']});\n    model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});\n    return await model.fit(xs, ys, {batchSize, epochs});\n}\n\n// Generate text\nfunction generateText(model, wordIndex, seed, length) {\n    let sequence = seed.split(' ').map(word => wordIndex[word]);\n    for (let i = 0; i < length; i++) {\n        const prediction = model.predict(tf.tensor2d(sequence, [1, sequence.length]));\n        sequence.push(prediction.argMax(-1).dataSync()[0]);\n    }\n    return sequence.map(index => Object.keys(wordIndex).find(word => wordIndex[word] === index)).join(' ');\n}\n\ndocument.getElementById('train').addEventListener('click', async () => {\n    const text = await loadData();\n    const { sequences } = preprocessData(text);\n    const model = buildModel(VOCAB_SIZE, EMBEDDING_SIZE, LSTM_UNITS);\n    tfvis.show.modelSummary({name: 'Layers'}, model);\n    await trainModel(model, sequences, BATCH_SIZE, EPOCHS);\n    model.save('localstorage://model');\n});\n\ndocument.getElementById('test').addEventListener('click', async () => {\n    const model = await tf.loadLayersModel('localstorage://model').catch(() => null);\n    if (!model) return document.getElementById('output').innerText = 'Error: No model trained.';\n    const wordIndex = preprocessData(await loadData()).wordIndex;\n    const seed = document.getElementById('input').value;\n    document.getElementById('output').innerHTML = generateText(model, wordIndex, seed, SEQ_LENGTH);\n});\n\n//# sourceURL=webpack:///./script.js?");

/***/ }),

/***/ "./tokenizer.js":
/*!**********************!*\
  !*** ./tokenizer.js ***!
  \**********************/
/***/ ((module) => {

eval("// https://gist.github.com/dlebech/5bbabaece36753f8a29e7921d8e5bfc7\nclass Tokenizer {\n  constructor(config = {}) {\n    this.filters = config.filters || /[\\\\.,/#!$%^&*;:{}=\\-_`~()]/g;\n    this.lower = typeof config.lower === 'undefined' ? true : config.lower;\n\n    // Primary indexing methods. Word to index and index to word.\n    this.wordIndex = {};\n    this.indexWord = {};\n\n    // Keeping track of word counts\n    this.wordCounts = {};\n  }\n\n  cleanText(text) {\n    if (this.lower) text = text.toLowerCase();\n    return text\n      .replace(this.filters, '')\n      .replace(/\\s{2,}/g, ' ')\n      .split(' ');\n  }\n\n  fitOnTexts(texts) {\n    texts.forEach(text => {\n      text = this.cleanText(text);\n      text.forEach(word => {\n        this.wordCounts[word] = (this.wordCounts[word] || 0) + 1;\n      });\n    });\n\n    Object.entries(this.wordCounts)\n      .sort((a, b) => b[1] - a[1])\n      .forEach(([word, number], i) => {\n        this.wordIndex[word] = i + 1;\n        this.indexWord[i + 1] = word;\n      });\n  }\n\n  textsToSequences(texts) {\n    return texts.map(text => this.cleanText(text).map(word => this.wordIndex[word] || 0));\n  }\n\n  toJson() {\n    return JSON.stringify({\n      wordIndex: this.wordIndex,\n      indexWord: this.indexWord,\n      wordCounts: this.wordCounts\n    })\n  }\n}\n\nconst tokenizerFromJson = json_string => {\n  const tokenizer = new Tokenizer();\n  const js = JSON.parse(json_string);\n  tokenizer.wordIndex = js.wordIndex;\n  tokenizer.indexWord = js.indexWord;\n  tokenizer.wordCounts = js.wordCounts;\n  return tokenizer;\n};\n\nmodule.exports = { Tokenizer, tokenizerFromJson };\n\n//# sourceURL=webpack:///./tokenizer.js?");

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module can't be inlined because the eval devtool is used.
/******/ 	var __webpack_exports__ = __webpack_require__("./script.js");
/******/ 	
/******/ })()
;