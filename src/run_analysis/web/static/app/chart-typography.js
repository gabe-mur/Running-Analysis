// SVG viewBox scaling must not turn readable axis labels into tiny text.
// Only presentation changes: data, geometry and hit-testing stay unchanged.
const observers = new WeakMap();

export function chartLabelSize(scale, pixels = 12) {
  return Number.isFinite(scale) && scale > 0 && Number.isFinite(pixels) && pixels > 0
    ? pixels / scale : null;
}

export function wireChartTypography(root) {
  observers.get(root)?.disconnect();
  observers.delete(root);
  if (typeof globalThis.ResizeObserver !== 'function') return;
  const charts = [...root.querySelectorAll('svg.fitness-chart, svg.vo2-chart, svg.insight-chart, svg.effort-history, svg.comparison-history, svg.comparison-range, svg.mileage-chart')];
  if (!charts.length) return;
  const update = svg => {
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const pixels = Number(getComputedStyle(svg).getPropertyValue('--chart-label-px').trim()) || 12;
    const size = chartLabelSize(Math.hypot(matrix.a, matrix.b), pixels);
    if (size !== null) svg.style.setProperty('--chart-label-size', `${size}px`);
  };
  const observer = new ResizeObserver(entries => entries.forEach(({target}) => update(target)));
  charts.forEach(svg => { update(svg); observer.observe(svg); });
  observers.set(root, observer);
}
