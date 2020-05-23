const puppeteer = require('puppeteer');
const fs = require('fs');
const https = require('https');

(async () => {
  const browser = await puppeteer.launch({headless: false});
  const page = await browser.newPage();
  await page.setViewport({width: 1200, height: 800});
  await page.goto('http://optc-db.github.io/characters/', {waitUntil: "networkidle0"});

  var has_next_page = true;
  var units = [];
  var index = 1;

  const saveImage = (url, filepath) => {
    console.log('Getting image from: ' + url);
    let request = https.get(url, response => {
      let file = fs.createWriteStream(filepath);
      response.pipe(file);
    }).on('error', e => {
      console.error('Count not find image at: ' + url);
    });
  }

  await page.select('select[name="mainTable_length"]', '100');

  while(has_next_page) {

    var new_units = await page.evaluate(unit_names => {
      scraped_units = [];
      document.querySelectorAll('#mainTable > tbody > tr').forEach(line => {
        let unit = {};
        unit.id = line.children[0].innerText.trim();
        unit.name = line.children[1].children[1].innerText.trim();
        unit.imgUrl = line.children[1].children[0].getAttribute('src').trim();
        scraped_units.push(unit);
      });

      return scraped_units;
    });

    units.push(...new_units);

    has_next_page = await page.evaluate(() => {
      return !document.querySelector('#mainTable_paginate .next').classList.contains('disabled');
    });

    if (has_next_page) {
      await page.click('#mainTable_paginate ul li:last-child a', {waitUntil: "networkidle0"});
      //await page.waitFor(500);
    }

  }

  console.log(units);

  units.forEach(unit => {
    let path = 'images/' + unit.id + '/';
    fs.mkdir(path, { recursive: true }, (err) => {
      if (err) throw err;
    })
    saveImage(unit.imgUrl, path + unit.id + '.png');
  });

  console.log(units);

  fs.writeFile('units.json', JSON.stringify(units), (err) => {
    if (err) throw err;
    console.log('File saved');
  });

  await browser.close();
})();


// Credit: Jihee Jeong
//          ()()
//          (oo)
//          ()()
//          (  )
//          ()()
