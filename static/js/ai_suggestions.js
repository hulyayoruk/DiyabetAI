// static/js/ai_suggestions.js
document.addEventListener("DOMContentLoaded", () => {
  const riskLabelEl = document.getElementById("ai-risk-label");
  const riskDescEl = document.getElementById("ai-risk-desc");
  const typeNoteEl = document.getElementById("ai-type-note");
  const simContainer = document.getElementById("ai-simulations");

  function fmtPct(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "--";
    if (n > 0 && n < 0.05) return "<0.1%";
    return `${n.toFixed(1)}%`;
  }

  function buildNoChangeText(before, after, key) {
    const b = Number(before?.[key]);
    const a = Number(after?.[key]);
    if (!Number.isFinite(b) || !Number.isFinite(a)) return "";
    const diff = Math.abs(a - b);

    // 0.05% altını “anlamlı değişim yok” say
    if (diff < 0.05) {
      return "Bu senaryoda olasılık üzerinde anlamlı bir fark oluşmadı.";
    }
    return "";
  }

  async function fetchAISuggestions() {
    try {
      const resp = await fetch("/api/ai_suggestions");
      const data = await resp.json();

      if (!resp.ok) {
        riskLabelEl.textContent = "AI önerileri şu an üretilemedi";
        riskDescEl.textContent = data.error || "Bilinmeyen hata";
        simContainer.style.display = "none";
        return;
      }

      renderRiskBlock(data);
      renderSimulations(data.simulations || []);
    } catch (err) {
      console.error("AI önerileri hatası:", err);
      riskLabelEl.textContent = "AI önerileri şu an üretilemedi";
      riskDescEl.textContent = "İstemci tarafında hata oluştu (Console'u kontrol et).";
      simContainer.style.display = "none";
    }
  }

  function renderRiskBlock(data) {
    riskLabelEl.textContent = data.risk_label || "";
    riskDescEl.textContent = data.risk_desc || "";
    typeNoteEl.textContent = "Bu analiz bir yapay zeka modelinin çıktısıdır.";

    const card = document.querySelector(".ai-main");
    if (card) {
      card.classList.remove("risk-low", "risk-normal", "risk-high");
      if ((data.risk_label || "").includes("Hipoglisemi")) card.classList.add("risk-low");
      else if ((data.risk_label || "").includes("Kontrol")) card.classList.add("risk-normal");
      else card.classList.add("risk-high");
    }

    // gece hipo satırı
    let nightEl = document.getElementById("night-hypo-line");
    if (!nightEl) {
      nightEl = document.createElement("p");
      nightEl.id = "night-hypo-line";
      nightEl.style.marginTop = "10px";
      nightEl.style.fontWeight = "600";
      riskDescEl.after(nightEl);
    }

    const dose = Number(data?.insulin_context?.current_insulin_dose ?? 0).toFixed(1);

    if (data.is_night) {
      nightEl.style.display = "block";
      nightEl.textContent = `🌙 Bu dozda gece hipo riski: %${Number(
        data.night_hypo_risk_pct || 0
      ).toFixed(1)} (Mevcut doz: ${dose}U)`;
    } else {
      nightEl.style.display = "none";
    }

    // hipo uyarı kutusu
    let warnBox = document.getElementById("hypo-warning-box");
    if (!warnBox) {
      warnBox = document.createElement("div");
      warnBox.id = "hypo-warning-box";
      warnBox.style.marginTop = "10px";
      warnBox.style.padding = "10px 12px";
      warnBox.style.borderRadius = "12px";
      warnBox.style.fontSize = "13px";
      riskDescEl.after(warnBox);
    }

    if (data.hypo_warning) {
      warnBox.style.display = "block";
      warnBox.innerHTML = `<strong>${data.hypo_warning.title}</strong><br>${data.hypo_warning.text}`;

      if (data.hypo_warning.level === "high") {
        warnBox.style.background = "#fee2e2";
        warnBox.style.border = "1px solid #fecaca";
        warnBox.style.color = "#991b1b";
      } else {
        warnBox.style.background = "#ffedd5";
        warnBox.style.border = "1px solid #fed7aa";
        warnBox.style.color = "#9a3412";
      }
    } else {
      warnBox.style.display = "none";
    }
  }

  function renderSimulations(simulations) {
    simContainer.innerHTML = "";

    if (!simulations.length) {
      simContainer.style.display = "none";
      return;
    }

    simContainer.style.display = "grid";

    simulations.forEach((sim) => {
      const before = sim.before || {};
      const after = sim.after || {};

      const hypoLine = `Hipoglisemi olasılığı: <strong>${fmtPct(before.p_hypo)} → ${fmtPct(after.p_hypo)}</strong>`;
      const hyperLine = `Hiperglisemi olasılığı: <strong>${fmtPct(before.p_hyper)} → ${fmtPct(after.p_hyper)}</strong>`;

      // “neden aynı?” açıklaması: iki satır da değişmiyorsa tek bir not göster
      const noteHypo = buildNoChangeText(before, after, "p_hypo");
      const noteHyper = buildNoChangeText(before, after, "p_hyper");
      const noChangeNote =
        noteHypo && noteHyper ? "Bu senaryoda olasılıklar üzerinde anlamlı bir fark oluşmadı." : "";

      const card = document.createElement("div");
      card.className = "ai-sim-card";
      card.innerHTML = `
        <p class="ai-sim-title">${sim.title || ""}</p>
        <p class="ai-sim-desc">${sim.subtitle || ""}</p>

        <p class="ai-sim-prob">${hypoLine}</p>
        <p class="ai-sim-prob">${hyperLine}</p>

        ${
          noChangeNote
            ? `<p class="ai-sim-note">${noChangeNote}</p>`
            : ""
        }
      `;
      simContainer.appendChild(card);

      colorSimCard(card, before, after);
    });
  }

  function colorSimCard(card, before, after) {
    // Renkleme: “en kötüleşen” risk artışına göre
    card.classList.remove("ai-risk-up", "ai-risk-down", "ai-risk-same");

    const bHypo = Number(before?.p_hypo);
    const aHypo = Number(after?.p_hypo);
    const bHyper = Number(before?.p_hyper);
    const aHyper = Number(after?.p_hyper);

    const dHypo = Number.isFinite(bHypo) && Number.isFinite(aHypo) ? aHypo - bHypo : 0;
    const dHyper = Number.isFinite(bHyper) && Number.isFinite(aHyper) ? aHyper - bHyper : 0;

    // En büyük mutlak değişimi baz al
    const d = Math.abs(dHypo) >= Math.abs(dHyper) ? dHypo : dHyper;

    if (d > 0.05) card.classList.add("ai-risk-up");
    else if (d < -0.05) card.classList.add("ai-risk-down");
    else card.classList.add("ai-risk-same");
  }

  fetchAISuggestions();
});
