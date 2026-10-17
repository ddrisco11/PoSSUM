/** Author the reviewed dataset in a workbook, verify values, then serialize CSV. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {Workbook} = await import(process.env.ARTIFACT_TOOL_ENTRY
  ? pathToFileURL(process.env.ARTIFACT_TOOL_ENTRY).href : '@oai/artifact-tool');
const root = process.argv[2] ?? 'data/PoSSUMDatasetv1.0';
const input = JSON.parse(await fs.readFile(path.join(root, 'workbook-payload.json'), 'utf8'));
const matrix = [input.columns, ...input.rows];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('PoSSUMDatasetv1.0');
const block = sheet.getRangeByIndexes(0, 0, matrix.length, input.columns.length);
block.values = matrix;
workbook.recalculate();
const verified = block.values;
assert.deepEqual(verified, matrix, 'Workbook values changed during authoring');
const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
const csv = '\ufeff' + verified.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
const dest = path.join(root, 'PoSSUMDatasetv1.0.csv');
await fs.writeFile(dest + '.tmp', csv, 'utf8');
await fs.rename(dest + '.tmp', dest);
console.log(JSON.stringify({path:dest, rows:input.rows.length, columns:input.columns.length, workbookRoundtrip:true}));

if (process.env.POSSUM_PREVIEW) {
  const preview = Workbook.create();
  const page = preview.worksheets.add('Preview');
  const fields = ['text','label_name','sub_dataset','quality_tier','partition'];
  const indices = fields.map(k => input.columns.indexOf(k));
  const sourceIndex = input.columns.indexOf('sub_dataset');
  const samples = [...new Set(input.rows.map(r => r[sourceIndex]))]
    .flatMap(source => input.rows.filter(r => r[sourceIndex] === source).slice(0, 2));
  const values = [fields, ...samples.map(r => indices.map(i => r[i]))];
  page.getRangeByIndexes(0,0,values.length,fields.length).values = values;
  page.getRange('A1:E7').format.wrapText = true;
  page.getRange('A:A').format.columnWidthPx = 580;
  page.getRange('B:B').format.columnWidthPx = 65;
  page.getRange('C:E').format.columnWidthPx = 180;
  page.getRange('A1:E1').format.fill = '#20364D';
  page.getRange('A1:E1').format.font = {bold:true,color:'#FFFFFF'};
  page.getRange('A2:E7').format.rowHeightPx = 130;
  preview.recalculate();
  const image = await preview.render({sheetName:'Preview',range:'A1:E7',scale:1,format:'png'});
  await fs.writeFile(process.env.POSSUM_PREVIEW, new Uint8Array(await image.arrayBuffer()));
}
