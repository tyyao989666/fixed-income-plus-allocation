import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [input, output] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
for (const [name, table] of Object.entries(data)) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  const rows = [table.columns, ...table.data.map(row => row.map((value, index) => {
    if (typeof value === 'string' && value.startsWith('=')) return `'${value}`;
    if (['date','start','end','start_date','end_date'].includes(table.columns[index]) && value && Number.isFinite(Date.parse(value))) return new Date(value);
    return value;
  }))];
  if (!rows[0].length) continue;
  const n = rows[0].length;
  sheet.getRangeByIndexes(0, 0, rows.length, n).values = rows;
  const used = sheet.getRangeByIndexes(0, 0, rows.length, n);
  used.format.font.name = 'Calibri';
  used.format.font.size = 10;
  used.format.columnWidth = 19;
  used.format.rowHeight = 19;
  const header = sheet.getRangeByIndexes(0, 0, 1, n);
  header.format.fill = '#176B65';
  header.format.font.color = '#FFFFFF';
  header.format.font.bold = true;
  header.format.wrapText = true;
  header.format.rowHeight = 44;
  sheet.freezePanes.freezeRows(1);
  table.columns.forEach((column, index) => {
    if (rows.length < 2) return;
    const cells = sheet.getRangeByIndexes(1, index, rows.length - 1, 1);
    if (['date','start','end','start_date','end_date'].includes(column)) cells.setNumberFormat('yyyy-mm-dd');
    if (/cagr|annual_volatility|max_drawdown|total_return|positive_fraction|linked_contribution/.test(column)) cells.setNumberFormat('0.00%');
    else if (/sharpe/.test(column)) cells.setNumberFormat('0.000');
    if (/path|value|reason|status|field/.test(column)) {
      sheet.getRangeByIndexes(0, index, rows.length, 1).format.columnWidth = 48;
      cells.format.wrapText = true;
    }
  });
  used.format.autofitRows();
  header.format.rowHeight = 44;
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'metrics!A1:J5',include:'values',tableMaxRows:5,tableMaxCols:10,maxChars:1600})).ndjson);
await fs.mkdir(path.dirname(output), {recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
for (const name of Object.keys(data)) {
  const preview = await wb.render({sheetName:name,range:'A1:F7',scale:1,format:'png'});
  await fs.writeFile(path.join(path.dirname(input),`${name}_preview.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(`Workbook exported: ${output}`);
