// static/js/measurement_report.js

document.addEventListener("DOMContentLoaded", () => {
  // HTML'deki id'lere göre:
  const daysSelect = document.getElementById("measurement-range");
  const tbody = document.getElementById("measurement-tbody");
  const btnExcel = document.getElementById("btn-export-excel");
  const btnPdf = document.getElementById("btn-export-pdf");

  if (!daysSelect || !tbody) {
    // Kart hiç yoksa sessizce çık
    return;
  }

  async function loadReport() {
    const days = daysSelect.value;

    tbody.innerHTML = `
      <tr>
        <td colspan="4" class="text-muted">Veri yükleniyor...</td>
      </tr>
    `;

    try {
      const resp = await fetch(`/api/measurement_list?days=${days}`);
      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }

      const data = await resp.json();

      if (!data.rows || data.rows.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="4" class="text-muted">
              Bu dönem için kayıt bulunamadı.
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = "";

      data.rows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.datetime || "-"}</td>
          <td>${row.glucose !== null && row.glucose !== undefined ? row.glucose : "-"}</td>
          <td>${row.drug || "-"}</td>
          <td>${row.note || "-"}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.error("Ölçüm raporu yüklenirken hata:", err);
      tbody.innerHTML = `
        <tr>
          <td colspan="4" class="text-danger">
            Rapor yüklenirken bir hata oluştu.
          </td>
        </tr>
      `;
    }
  }

  // Seçim değişince yeniden yükle
  daysSelect.addEventListener("change", loadReport);

  // Excel indirme
  if (btnExcel) {
    btnExcel.addEventListener("click", (e) => {
      e.preventDefault();
      const days = daysSelect.value;
      window.location.href = `/export_measurements_excel?days=${days}`;
    });
  }

  // PDF indirme
  if (btnPdf) {
    btnPdf.addEventListener("click", (e) => {
      e.preventDefault();
      const days = daysSelect.value;
      window.location.href = `/export_measurements_pdf?days=${days}`;
    });
  }

  // İlk yükleme
  loadReport();
});
