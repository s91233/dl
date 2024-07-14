// JS port of Keras Tokenizer by Copilot <3
function pad_sequences(sequences, maxlen=null, dtype='int32', padding='pre', truncating='pre', value=0.0) {
    if (!Array.isArray(sequences)) {
        throw new Error("`sequences` must be iterable.");
    }
    let num_samples = sequences.length;

    let lengths = [];
    let sample_shape = [];
    let flag = true;

    for (let x of sequences) {
        try {
            lengths.push(x.length);
            if (flag && x.length) {
                sample_shape = [x.length];
                flag = false;
            }
        } catch (e) {
            throw new Error("`sequences` must be a list of iterables. Found non-iterable: " + x);
        }
    }

    if (maxlen === null) {
        maxlen = Math.max(...lengths);
    }

    let x = Array(num_samples).fill().map(() => Array(maxlen).fill(value));

    for (let idx = 0; idx < sequences.length; idx++) {
        let s = sequences[idx];
        if (!s.length) {
            continue;
        }
        let trunc;
        if (truncating === 'pre') {
            trunc = s.slice(-maxlen);
        } else if (truncating === 'post') {
            trunc = s.slice(0, maxlen);
        } else {
            throw new Error('Truncating type "' + truncating + '" not understood');
        }

        if (padding === 'post') {
            x[idx].splice(0, trunc.length, ...trunc);
        } else if (padding === 'pre') {
            x[idx].splice(-trunc.length, trunc.length, ...trunc);
        } else {
            throw new Error('Padding type "' + padding + '" not understood');
        }
    }
    return x;
}

async function plot_history(canvas, path) {
    const response = await fetch(path + '/history.json');
    const history = await response.json();

    const metrics = Object.keys(history.history);

    if (window.histplot) {
        window.histplot.destroy();
    }

    window.histplot = new Chart(canvas, {
        type: 'line',
        data: {
            labels: Array.from({ length: history.epoch.length }, (_, i) => i + 1),
            datasets: metrics.map(metric => ({
                label: metric, data: history.history[metric],
            })),
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'Model Training History',
            },
            scales: {
                x: {
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Epoch',
                    },
                },
                y: {
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Value',
                    },
                    type: 'logarithmic',
                },
            },
        },
    });
}

module.exports = { pad_sequences, plot_history };
