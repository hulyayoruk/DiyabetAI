document.addEventListener("DOMContentLoaded", async () => {
  const canvas = document.getElementById("glucoseChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let glucoseChart = null;

  async function fetchData() {
    try {
      const response = await fetch("/api/data");
      const data = await response.json();

      const values = data.values || [];
      const labels = data.labels || [];

      // 🔹 Anlık ve ortalama değerleri güncelle
      if (values.length > 0) {
        const current = values[values.length - 1];
        const avg =
          values.reduce((a, b) => a + b, 0) / values.length;

        const currentEl = document.getElementById("current");
        const avgEl = document.getElementById("average");

        if (currentEl) currentEl.textContent = `${current} mg/dL`;
        if (avgEl) avgEl.textContent = `${avg.toFixed(1)} mg/dL`;
      }

      // 🔹 Y ekseni için dinamik min–max (veriye göre)
      let yMin = 50;
      let yMax = 300;
      if (values.length > 0) {
        const vMin = Math.min(...values);
        const vMax = Math.max(...values);
        const padding = 20;

        yMin = Math.max(0, Math.floor((vMin - padding) / 10) * 10);
        yMax = Math.ceil((vMax + padding) / 10) * 10;
      }

      if (glucoseChart) {
        // Mevcut grafiği güncelle
        glucoseChart.data.labels = labels;
        glucoseChart.data.datasets[0].data = values;
        glucoseChart.options.scales.y.min = yMin;
        glucoseChart.options.scales.y.max = yMax;
        glucoseChart.update();
      } else {
        // Yeni grafik oluştur
        glucoseChart = new Chart(ctx, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Kan Şekeri (mg/dL)",
                data: values,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.15)",
                borderWidth: 2,
                tension: 0.35,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 5
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false, // kartın yüksekliğini daha iyi doldursun
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx) => `${ctx.parsed.y} mg/dL`
                }
              }
            },
            scales: {
              x: {
                ticks: {
                  maxRotation: 45,
                  minRotation: 45
                }
              },
              y: {
                min: yMin,
                max: yMax,
                ticks: {
                  stepSize: 20,
                  callback: (value) => `${value}`
                }
              }
            }
          }
        });
      }
    } catch (err) {
      console.error("Grafik verisi alınırken hata:", err);
    }
  }
  await fetchData();
  // Her 60 sn’de bir güncelle
  setInterval(fetchData, 60000);

  // Ekran boyutu değişirse grafiği yeniden boyutlandır
  window.addEventListener("resize", () => {
    if (glucoseChart) glucoseChart.resize();
  });
});

