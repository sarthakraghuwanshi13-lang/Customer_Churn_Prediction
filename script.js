/**
 * ChurnSense AI — Frontend Logic
 * Handles form submission, API communication, and result animation.
 */

const API_BASE = "http://127.0.0.1:5000";

// ─── DOM Elements ───────────────────────────────────────────
const predictForm = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const btnText = predictBtn.querySelector(".btn-text");
const btnLoader = predictBtn.querySelector(".btn-loader");
const resultCard = document.getElementById("result-card");
const closeResult = document.getElementById("close-result");
const modelAccuracy = document.getElementById("model-accuracy");

// ─── Load Model Info ────────────────────────────────────────
async function loadModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/model-info`);
        const data = await res.json();
        modelAccuracy.textContent = `Model Accuracy: ${(data.test_accuracy * 100).toFixed(1)}%`;
    } catch (err) {
        modelAccuracy.textContent = "Model: Ready";
    }
}

loadModelInfo();

// ─── Form Submit ────────────────────────────────────────────
predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    // Gather form data
    const formData = new FormData(predictForm);
    const payload = {};
    
    for (const [key, value] of formData.entries()) {
        // Convert numeric fields
        if (["tenure", "SeniorCitizen"].includes(key)) {
            payload[key] = parseInt(value);
        } else if (["MonthlyCharges", "TotalCharges"].includes(key)) {
            payload[key] = parseFloat(value);
        } else {
            payload[key] = value;
        }
    }

    // UI: loading state
    btnText.style.display = "none";
    btnLoader.style.display = "inline-flex";
    predictBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        showResult(data);

    } catch (err) {
        alert("Error: " + err.message + "\n\nMake sure the Flask server is running on port 5000.");
    } finally {
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
        predictBtn.disabled = false;
    }
});

// ─── Show Result ────────────────────────────────────────────
function showResult(data) {
    resultCard.classList.remove("hidden");
    
    // Scroll to result
    setTimeout(() => {
        resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 100);

    const churnProb = data.probability.churn;
    const noChurnProb = data.probability.no_churn;
    const isChurn = data.churn;

    // Prediction label
    const predLabel = document.getElementById("prediction-label");
    const predIcon = document.getElementById("pred-icon");
    const predText = document.getElementById("pred-text");

    predLabel.className = "prediction-label " + (isChurn ? "risk" : "safe");
    predIcon.textContent = isChurn ? "⚠️" : "✅";
    predText.textContent = isChurn ? "High Churn Risk" : "Customer is Safe";

    // Gauge animation
    const gaugeFill = document.getElementById("gauge-fill");
    const gaugeValue = document.getElementById("gauge-value");

    // Total arc length ≈ 251.33 for d="M 20 100 A 80 80 0 0 1 180 100"
    const totalArc = 251.33;
    const targetOffset = totalArc * (1 - churnProb);

    // Set gauge color
    if (isChurn) {
        gaugeFill.setAttribute("stroke", "url(#gaugeGradRed)");
        gaugeValue.style.color = "var(--red)";
    } else {
        gaugeFill.setAttribute("stroke", "url(#gaugeGradGreen)");
        gaugeValue.style.color = "var(--green)";
    }

    // Animate gauge
    gaugeFill.style.strokeDashoffset = totalArc; // reset
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            gaugeFill.style.strokeDashoffset = targetOffset;
        });
    });

    // Animate counter
    animateCounter(gaugeValue, 0, Math.round(churnProb * 100), 1200, "%");

    // Probability bars
    const probBarSafe = document.getElementById("prob-bar-safe");
    const probBarRisk = document.getElementById("prob-bar-risk");
    const probSafe = document.getElementById("prob-safe");
    const probRisk = document.getElementById("prob-risk");

    probBarSafe.style.width = "0%";
    probBarRisk.style.width = "0%";

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            probBarSafe.style.width = `${noChurnProb * 100}%`;
            probBarRisk.style.width = `${churnProb * 100}%`;
        });
    });

    probSafe.textContent = `${(noChurnProb * 100).toFixed(1)}%`;
    probRisk.textContent = `${(churnProb * 100).toFixed(1)}%`;

    // Confidence
    document.getElementById("confidence-val").textContent = `${data.confidence}%`;
}

// ─── Animate Counter ────────────────────────────────────────
function animateCounter(element, from, to, duration, suffix = "") {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(from + (to - from) * eased);
        
        element.textContent = current + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ─── Close Result ───────────────────────────────────────────
closeResult.addEventListener("click", () => {
    resultCard.classList.add("hidden");
});
