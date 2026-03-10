// static/js/chart.js
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("glucoseChart");
  const trendPill = document.getElementById("glucose-trend-pill");

  if (!canvas) return;

  let chartInstance = null;

  function setTrendPill(trendText, trendClass) {
    if (!trendPill) return;

    // metin
    trendPill.textContent = trendText || "Normal / stabil";

    // class reset + set
    trendPill.classList.remove("trend-up", "trend-down", "trend-neutral");

    if (trendClass === "up") trendPill.classList.add("trend-up");
    else if (trendClass === "down") trendPill.classList.add("trend-down");
    else trendPill.classList.add("trend-neutral");
  }

  async function loadChart() {
    try {
      const resp = await fetch("/api/data");
      const data = await resp.json();

      if (!resp.ok || data.error) {
        setTrendPill("Trend yok", "neutral");
        return;
      }

      // ✅ Trend etiketi (dinamik)
      setTrendPill(data.trend_text, data.trend_class);

      const labels = data.labels || [];
      const values = data.values || [];

      if (!window.Chart) return;

      if (chartInstance) {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = values;
        chartInstance.update();
        return;
      }

      chartInstance = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Glikoz",
              data: values,
              fill: true,
              tension: 0.35,
              borderWidth: 2,
              pointRadius: 3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { enabled: true },
          },
          scales: {
            y: { beginAtZero: false },
          },
        },
      });
    } catch (err) {
      console.error("chart load error:", err);
      setTrendPill("Trend yüklenemedi", "neutral");
    }
  }

  loadChart();

  // İstersen canlı güncelleme:
  // setInterval(loadChart, 2 * 60 * 1000);
});
