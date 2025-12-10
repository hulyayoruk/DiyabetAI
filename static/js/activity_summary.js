document.addEventListener("DOMContentLoaded", async () => {
  // Kart elemanları
  const elCarb  = document.getElementById("sum_carbs");
  const elEx    = document.getElementById("sum_ex");
  const elInt   = document.getElementById("avg_int");
  const elSteps = document.getElementById("sum_steps");
  const elSleep = document.getElementById("sum_sleep");

  if (!elCarb || !elEx || !elInt || !elSteps || !elSleep) {
    // Kart yoksa sessizce çık
    return;
  }

  async function loadSummary() {
    try {
      const res = await fetch("/api/activity_summary");
      if (!res.ok) {
        console.error("Özet API hatası:", res.status);
        return;
      }
      const data = await res.json();
      if (data.error) {
        console.warn("API hatası:", data.error);
        return;
      }

      elCarb.textContent  = `${data.total_carb.toFixed(1)} g`;
      elEx.textContent    = `${data.total_ex_minutes.toFixed(0)} dk`;
      elInt.textContent   = `${data.avg_intensity.toFixed(1)} / 5`;
      elSteps.textContent = data.total_steps.toString();
      elSleep.textContent = `${data.total_sleep.toFixed(0)} dk`;
    } catch (err) {
      console.error("activity_summary istek hatası:", err);
    }
  }

  // İlk yüklemede getir
  await loadSummary();

  // İstersen 5 dakikada bir yenile:
  setInterval(loadSummary, 5 * 60 * 1000);
});
