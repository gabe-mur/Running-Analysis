// Equipment guidance, not another setting or a brand-based compatibility claim.
export function setupDeviceCard() {
  return `<article class="wide-card setup-block" id="setup-device">
    <h2>What you’ll need</h2>
    <p>A GPS-capable watch or fitness tracker that records heart rate throughout your running workouts.</p>
    <p>We recommend an external heart-rate monitor, preferably a chest strap, for more consistent readings. It’s optional. Pair it with the watch or app recording your run so its readings are saved with the workout.</p>
    <details>
      <summary>Connecting your device</summary>
      <p><b>Connected watches:</b> Connect your watch service in Intervals.icu, then use our Sync button. Sync requires a downloadable original FIT file.</p>
      <p><b>Apple Watch:</b> Use the free <a href="https://apps.apple.com/us/app/intervals-icu-companion/id6739638454" target="_blank" rel="noopener noreferrer">Intervals.icu Companion</a> Apple Health bridge. Sign in to Intervals and allow workouts, heart rate and workout routes, then use our Sync button. We still need to verify an Apple Watch file end to end before confirming support here.</p>
      <p>Apple Health access is granted through an iPhone app, not the Intervals.icu website alone.</p>
      <p><b>Manual upload:</b> Upload FIT, TCX or GPX activity files. Garmin users can also upload a Garmin ZIP. Prefer the original FIT when available. GPX must contain workout timestamps; heart rate is needed for fitness analysis, and isn’t included in every GPX.</p>
      <p>Use normal workout recording, rather than settings that reduce GPS or heart-rate sampling. Extra watch metrics such as VO₂max and HRV aren’t required.</p>
    </details>
  </article>`;
}
