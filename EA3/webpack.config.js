// webpack.config.js
const path = require('path');

module.exports = {
    entry: './script.js',
    output: {
        filename: 'index.js', path: path.resolve(__dirname),
    },
};
