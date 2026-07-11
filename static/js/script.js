const TIER_COLORS = {
  "Low": "#B5473C",
  "Medium": "#C98A3A",
  "High": "#5C9C82",
  "Very High": "#3E7A5E",
};

const TIER_NOTES = {
  "Low": "Suggests significant gaps in health, education, or income that would benefit from targeted development investment.",
  "Medium": "A developing profile — meaningful gains are achievable with continued investment in one or more dimensions.",
  "High": "Strong fundamentals across health, education, and income, approaching the top development tier.",
  "Very High": "Among the strongest development profiles — long lifespans, deep educational attainment, and high income.",
};

const ARC_LENGTH = 282.6; // matches the SVG path's approximate arc length

// --- Sync number inputs <-> range sliders ---
document.querySelectorAll(".slider").forEach((slider) => {
  const targetId = slider.dataset.target;
  const numberInput = document.getElementById(targetId);

  slider.addEventListener("input", () => {
    numberInput.value = slider.value;
  });
  numberInput.addEventListener("input", () => {
    slider.value = numberInput.value;
  });
});

// --- Sample buttons ---
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    const tier = chip.dataset.tier;
    try {
      const res = await fetch(`/api/sample/${tier}`);
      const data = await res.json();
      if (data.error) return;
      Object.entries(data).forEach(([key, value]) => {
        const input = document.getElementById(key);
        const slider = document.querySelector(`.slider[data-target="${key}"]`);
        if (input) input.value = value;
        if (slider) slider.value = value;
      });
    } catch (err) {
      console.error("Failed to load sample:", err);
    }
  });
});

// --- Form submission ---
const form = document.getElementById("hdi-form");
const errorEl = document.getElementById("form-error");
const emptyState = document.getElementById("result-empty");
const contentState = document.getElementById("result-content");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;

  const payload = {
    life_expectancy: parseFloat(document.getElementById("life_expectancy").value),
    mean_years_schooling: parseFloat(document.getElementById("mean_years_schooling").value),
    expected_years_schooling: parseFloat(document.getElementById("expected_years_schooling").value),
    gni_per_capita: parseFloat(document.getElementById("gni_per_capita").value),
  };

  const submitBtn = form.querySelector(".submit-btn");
  const originalText = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = "<span>Reading&hellip;</span>";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorEl.textContent = data.error || "Something went wrong. Please check your inputs.";
      errorEl.hidden = false;
      return;
    }

    renderResult(data);
  } catch (err) {
    errorEl.textContent = "Could not reach the prediction service. Please try again.";
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalText;
  }
});

function renderResult(data) {
  emptyState.hidden = true;
  contentState.hidden = false;

  const color = TIER_COLORS[data.prediction] || "#C9A227";

  // Dial arc: map HDI score (0-1) to the arc fill
  const arc = document.getElementById("dial-arc");
  const offset = ARC_LENGTH * (1 - data.computed_hdi_score);
  arc.style.stroke = color;
  arc.style.strokeDashoffset = offset;

  // Needle: map 0-1 to -90deg .. +90deg
  const needle = document.getElementById("dial-needle");
  const angle = -90 + data.computed_hdi_score * 180;
  needle.style.transform = `rotate(${angle}deg)`;

  document.getElementById("hdi-score").textContent = data.computed_hdi_score.toFixed(3);

  const tierBadge = document.getElementById("tier-badge");
  const tierLabel = document.getElementById("tier-label");
  tierLabel.textContent = data.prediction;
  tierBadge.querySelector("span").style.borderColor = color;
  tierBadge.querySelector("span").style.color = color;

  document.getElementById("confidence-value").textContent =
    (data.confidence * 100).toFixed(1) + "%";

  document.getElementById("tier-note").textContent = TIER_NOTES[data.prediction] || "";

  // Probability bars, ordered Low -> Very High
  const order = ["Low", "Medium", "High", "Very High"];
  const barsContainer = document.getElementById("prob-bars");
  barsContainer.innerHTML = "";
  order.forEach((tier) => {
    const p = data.probabilities[tier] ?? 0;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${tier}</span>
      <span class="prob-track"><span class="prob-fill" style="width:0%; background:${TIER_COLORS[tier]}"></span></span>
      <span class="prob-value">${(p * 100).toFixed(1)}%</span>
    `;
    barsContainer.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${p * 100}%`;
    });
  });
}
