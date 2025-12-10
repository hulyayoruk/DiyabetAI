// static/js/period_report.js

document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("period-chart");
  if (!ctx) return;

  let periodChart = null;

  const selectEl = document.getElementById("period-select");
  const avgEl = document.getElementById("pr_avg_glucose");
  const inrangeEl = document.getElementById("pr_inrange");
  const hypoEl = document.getElementById("pr_hypo");
  const hyperEl = document.getElementById("pr_hyper");
  const deltaEl = document.getElementById("pr_delta");

  // Donut chart oluştur
  function createChart() {
    periodChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Aralıkta (70–180)", "Hipoglisemi (< 70)", "Hiperglisemi (> 180)"],
        datasets: [
          {
            data: [0, 0, 0],
            backgroundColor: ["#34a853", "#fbbc05", "#ea4335"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",
        animation: {
          duration: 800,              // 🔥 her update’te belirgin animasyon
          easing: "easeOutQuad",
        },
        transitions: {
          active: {
            animation: {
              duration: 800,
              easing: "easeOutQuad",
            },
          },
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                const label = context.label || "";
                const value = context.parsed;
                return `${label}: ${value.toFixed(1)}%`;
              },
            },
          },
        },
      },
    });
  }

  function updateChart(inrange, hypo, hyper) {
    if (!periodChart) return;
    periodChart.data.datasets[0].data = [inrange, hypo, hyper];
    // 'active' transition ile animasyonlu güncelle
    periodChart.update("active");
  }

  function formatPct(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return `${v.toFixed(0)}%`;
  }

  function formatGlucose(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return `${v.toFixed(1)} mg/dL`;
  }

  function buildDeltaText(days, direction, delta) {
    const dayText = `${days} günlük dönem`;
    if (!direction || direction === "none" || delta === null) {
      return `${dayText}, bir önceki aynı döneme göre belirgin değişim göstermiyor.`;
    }

    const diffAbs = Math.abs(delta).toFixed(1);

    if (direction === "down") {
      return `${dayText}, önceki döneme göre ortalama kan şekeri ${diffAbs} mg/dL daha iyi görünüyor 🎉`;
    }
    if (direction === "up") {
      return `${dayText}, önceki döneme göre ortalama kan şekeri ${diffAbs} mg/dL daha yüksek görünüyor ⚠️`;
    }
    return `${dayText}, bir önceki aynı döneme göre belirgin değişim göstermiyor.`;
  }

  async function loadPeriodReport(days) {
    try {
      const resp = await fetch(`/api/period_report?days=${days}`);
      const data = await resp.json();

      if (!resp.ok || data.error) {
        // Yeterli veri yoksa
        avgEl.textContent = "--";
        inrangeEl.textContent = "--";
        hypoEl.textContent = "--";
        hyperEl.textContent = "--";

        updateChart(0, 0, 0);

        deltaEl.className = "period-delta-text period-delta-same";
        deltaEl.textContent = `${days} günlük dönem için yeterli veri yok.`;
        return;
      }

      const avg = data.avg_glucose;
      const inrange = data.inrange_pct;
      const hypo = data.hypo_pct;
      const hyper = data.hyper_pct;

      avgEl.textContent = formatGlucose(avg);
      inrangeEl.textContent = formatPct(inrange);
      hypoEl.textContent = formatPct(hypo);
      hyperEl.textContent = formatPct(hyper);

      updateChart(inrange, hypo, hyper);

      const direction = data.delta_direction;
      const delta = data.delta_avg;

      deltaEl.textContent = buildDeltaText(days, direction, delta);

      if (direction === "down") {
        deltaEl.className = "period-delta-text period-delta-better";
      } else if (direction === "up") {
        deltaEl.className = "period-delta-text period-delta-worse";
      } else {
        deltaEl.className = "period-delta-text period-delta-same";
      }
    } catch (err) {
      console.error("Period report fetch error:", err);
      avgEl.textContent = "--";
      inrangeEl.textContent = "--";
      hypoEl.textContent = "--";
      hyperEl.textContent = "--";

      updateChart(0, 0, 0);

      deltaEl.className = "period-delta-text period-delta-same";
      deltaEl.textContent = "Şu anda dönemsel özet yüklenemedi.";
    }
  }

  // İlk grafik oluşturma
  createChart();

  // Select değiştiğinde yeni dönem yükle
  if (selectEl) {
    selectEl.addEventListener("change", function () {
      const days = parseInt(this.value, 10) || 7;
      loadPeriodReport(days);
    });

    // İlk yükleme
    const initialDays = parseInt(selectEl.value, 10) || 7;
    loadPeriodReport(initialDays);
  }
});
