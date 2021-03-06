const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const https = require('https');
const fetch = require('node-fetch');

const saveImage = async (url, filepath) => {
  if (url.slice(0, 8) != 'https://') {
    console.log('Invalid url: ', url);
    return;
  }

  console.log('Getting image from: ' + url);



  await https.get(url, response => {
    let file = fs.createWriteStream(filepath);
    response.pipe(file);
  }).on('error', e => {
    throw e;
  });
}

const fetchImage = async (url, filepath) => {
  if (url.slice(0, 8) != 'https://') {
    console.log('Invalid url: ', url);
    return;
  }

  console.log('Getting image from: ' + url);

  return fetch(url)
  .then(response => response.buffer())
  .then(async (buffer) => {
    await fs.writeFile(filepath, buffer);
  });
}


(async () => {
  try {
    const browser = await puppeteer.launch({headless: true});
    const page = await browser.newPage();
    await page.setViewport({width: 1200, height: 800});
    await page.goto('http://optc-db.github.io/characters/', {waitUntil: "networkidle0"});

    var has_next_page = true;
    var units = [];
    var index = 1;

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

      // Wait 1 second
      //await page.waitFor(500);

      if (has_next_page) {
        await page.click('#mainTable_paginate ul li:last-child a', {waitUntil: "networkidle0"});
        //await page.waitFor(500);
      }

    }

    await browser.close();

    console.log(units);

    units.forEach(unit => {
      let path = 'images/';
      fs.mkdir(path, { recursive: true }, (err) => {
        if (err) throw err;
      })
      fetchImage(unit.imgUrl, path + unit.id + '.png');
    });

    console.log(units);

    fs.writeFile('units.json', JSON.stringify(units), (err) => {
      if (err) throw err;
      console.log('File saved');
    });
  }
  catch (err) {
    throw err;
  }
})();


// Credit: Jihee Jeong
//          ()()
//          (oo)
//          ()()
//          (  )
//          ()()
