// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Multiple Image classification using MobileNet and p5.js
=== */

// Initialize the Image Classifier method using MobileNet
let classifier;

let img = null;

function preload() {
    classifier = ml5.imageClassifier("MobileNet")
}

function setup() {
    createCanvas(512,512).parent('canvas').drop(drop)
}

function draw() {
    background('gray');
    textAlign(CENTER,CENTER);
    if (img) image(img,0,0);
    else text('Drop or select image',256,256);
}

function drop(file) {
    img = createImg(file.data,'').hide();
}

function upload() {
    const file = event.target.files[0];
    var reader = new FileReader();
    reader.onload = e => {
      if (file.type === 'image/png' || file.type === 'image/jpeg')
        img = createImg(e.target.result,'').hide();
    }; reader.readAsDataURL(file);
}

async function classify() {
    table = document.querySelector("tbody");
    for (i = 0; i < table.rows.length; i++) {
        row = table.rows[i];
        try {
            if (i===table.rows.length-1 && img) probe = img
            else
            probe = row.cells[0].querySelector("img");
            result = await classifier.classify(probe);
            console.log(`Image ${i+1} classification:`,result);

            Plotly.newPlot(row.cells[1], [{
                type: 'bar',
                x: result.map((p) => p.label),
                y: result.map((p) => p.confidence),
            }], {
                xaxis: {title: 'Class Label'},
                yaxis: {title: 'Confidence Score'},
                title: `Image ${i + 1} Prediction`,
            });
        } catch (e) {
            console.error(e);
        }
    }
}
