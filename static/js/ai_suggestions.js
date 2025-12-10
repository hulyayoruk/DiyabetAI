document.addEventListener("DOMContentLoaded", () => {
  const riskLabelEl = document.getElementById("ai-risk-label");
  const riskDescEl = document.getElementById("ai-risk-desc");
  const typeNoteEl = document.getElementById("ai-type-note");
  const simContainer = document.getElementById("ai-simulations");

  async function fetchAISuggestions() {
    try {
      const resp = await fetch("/api/ai_suggestions");
      if (!resp.ok) throw new Error("AI yanıtı alınamadı");

      const data = await resp.json();
      renderRiskBlock(data);
      renderSimulations(data.simulations || []);
    } catch (err) {
      console.error("AI önerileri hatası:", err);
    }
  }

  function renderRiskBlock(data) {
    riskLabelEl.textContent = data.risk_label;
    riskDescEl.textContent = data.risk_desc;
    typeNoteEl.textContent = "Bu analiz bir yapay zeka modelinin çıktısıdır.";
  }

  function renderSimulations(simulations) {
    simContainer.innerHTML = "";
    if (!simulations.length) {
      simContainer.style.display = "none";
      return;
    }

    simContainer.style.display = "grid";

    simulations.forEach(sim => {
      const card = document.createElement("div");
      card.className = "ai-sim-card";

      card.innerHTML = `
        <p class="ai-sim-title">${sim.title}</p>
        <p class="ai-sim-desc">${sim.subtitle}</p>
        <p class="ai-sim-prob">
          Hiperglisemi olasılığı:
          <strong>${sim.before_prob.toFixed(1)}% → ${sim.after_prob.toFixed(1)}%</strong>
        </p>
      `;

      simContainer.appendChild(card);
    });

    colorSimCardsByChange();
  }

  function colorSimCardsByChange() {
    document.querySelectorAll(".ai-sim-card").forEach(card => {
      const vals = card.innerText.match(/(\d+\.\d+)% → (\d+\.\d+)%/);
      if (!vals) return;

      const before = parseFloat(vals[1]);
      const after = parseFloat(vals[2]);

      if (after > before) card.classList.add("ai-risk-up");
      else if (after < before) card.classList.add("ai-risk-down");
      else card.classList.add("ai-risk-same");
    });
  }
  function renderRiskBlock(data) {
  riskLabelEl.textContent = data.risk_label;
  riskDescEl.textContent = data.risk_desc;
  typeNoteEl.textContent = "Bu analiz bir yapay zeka modelinin çıktısıdır.";

  const card = document.querySelector(".ai-main");

  card.classList.remove("risk-low", "risk-normal", "risk-high");

  if (data.risk_label.includes("Hipoglisemi")) {
    card.classList.add("risk-low");
  } else if (data.risk_label.includes("Kontrol")) {
    card.classList.add("risk-normal");
  } else {
    card.classList.add("risk-high");
  }
}


  fetchAISuggestions();
});
