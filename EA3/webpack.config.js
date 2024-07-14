const path = require('path');

module.exports = {
    entry: './script.js',
    output: {
        filename: 'index.js', path: path.resolve(__dirname),
    },
    devServer: {
        port: 8000, static: { directory: path.join(__dirname) },
    },
    devtool: 'source-map'
};
