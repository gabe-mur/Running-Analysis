import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chartLabelSize, wireChartTypography} from '../../src/run_analysis/web/static/app/chart-typography.js';

test('chart labels compensate for scaling without changing chart coordinates', () => {
  assert.equal(chartLabelSize(.25), 48);
  assert.equal(chartLabelSize(2), 6);
  assert.equal(chartLabelSize(.5, 14), 28);
  for (const scale of [0, -1, NaN, Infinity]) assert.equal(chartLabelSize(scale), null);
});

test('resize updates typography and rebinding disconnects the old observer', () => {
  const oldObserver = globalThis.ResizeObserver, oldStyle = globalThis.getComputedStyle;
  const instances = [];
  globalThis.ResizeObserver = class {
    constructor(callback) { this.callback=callback; this.targets=[]; instances.push(this); }
    observe(target) { this.targets.push(target); }
    disconnect() { this.disconnected=true; }
  };
  globalThis.getComputedStyle = () => ({getPropertyValue: () => '12'});
  let scale=.25;
  const properties = new Map();
  const svg = {getScreenCTM: () => ({a:scale,b:0}),
    style:{setProperty:(key,value)=>properties.set(key,value)}};
  let charts=[svg];
  const root={querySelectorAll:()=>charts};
  try {
    wireChartTypography(root);
    assert.equal(properties.get('--chart-label-size'), '48px');
    scale=.5;
    instances[0].callback([{target:svg}]);
    assert.equal(properties.get('--chart-label-size'), '24px');
    wireChartTypography(root);
    assert.ok(instances[0].disconnected);
    assert.deepEqual(instances[1].targets,[svg]);
    charts=[];
    wireChartTypography(root);
    assert.ok(instances[1].disconnected);
  } finally {
    globalThis.ResizeObserver=oldObserver; globalThis.getComputedStyle=oldStyle;
  }
});
