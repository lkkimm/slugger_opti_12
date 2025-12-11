async function run() {
  const pitcher_hand = document.getElementById("pitcher_hand").value;
  const batter_select = document.getElementById("batter_id");
  const batter_id = batter_select.value;
  const selected_option = batter_select.selectedOptions[0];
  const batter_name = selected_option?.dataset.name || "";
  const batter_hand = selected_option?.dataset.hand || "R";

  console.log("Selected batter info:", {
    batter_id,
    batter_name,
    batter_hand,
    selected_option,
    dataset: selected_option?.dataset
  });

  const button = document.querySelector('button[onclick="run()"]');
  const screen = document.getElementById("screen");

  // Disable button but do NOT show spinner
  button.disabled = true;
  button.textContent = "Generating...";

  try {
    const res = await fetch("/api/compute", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ 
        batter_id,
        pitcher_hand,
        batter_name,
        batter_hand
      })
    });

    const data = await res.json();

    if (!data.ok) {
      const errorMessage = data.error || "No data available for this player.";
      alert(errorMessage);

      screen.innerHTML = `
        <div class="initial-message">
          Select pitcher handedness and batter to generate optimized outfield positioning.
        </div>
      `;
      return;
    }

    const img = `<img class="tv" src="data:image/png;base64,${data.image_base64}" />`;
    const coords = Object.entries(data.positions)
      .map(([f, [x, y]]) => `${f}: X=${x.toFixed(1)}, Y=${y.toFixed(1)}`)
      .join(" • ");

    const warningMsg = data.warning
      ? `<div class="warning-message">⚠️ ${data.warning}</div>`
      : "";

    const caption = `
      <div class="caption">
        <span class="name">${data.batter_label}</span>
        &nbsp;&nbsp;vs.&nbsp;${data.pitcher_hand}
      </div>
      <div class="coords">${coords}</div>
      ${warningMsg}
    `;

    screen.innerHTML = img + caption;

  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred: " + error.message);

    screen.innerHTML = `
      <div class="initial-message">
        Select pitcher handedness and batter to generate optimized outfield positioning.
      </div>
    `;
  } finally {
    // Re-enable button
    button.disabled = false;
    button.textContent = "Generate Positions";
  }
}
