const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const fs = require('fs');
(async () => {
  const [svgPath, pngPath, w, h] = process.argv.slice(2);
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  const page = await browser.newPage({ viewport: { width: +w, height: +h } });
  const svg = fs.readFileSync(svgPath, 'utf8');
  await page.setContent(`<html><body style="margin:0">${svg.replace(/<svg([^>]*?) width="\d+" height="\d+"/, `<svg$1 width="${w}" height="${h}"`)}</body></html>`);
  await page.screenshot({ path: pngPath });
  await browser.close();
})();
