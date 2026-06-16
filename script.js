/* ================================================================
   LungCheck — Application Logic
   Sends form data to Flask API → displays real model predictions.
   ================================================================ */

const API_BASE = 'http://localhost:5000';

// ── DOM References ──────────────────────────────────────────────
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultsDiv = document.getElementById('results');

// ── Feature Toggle IDs (matches the 13 binary fields in HTML) ──
const TOGGLE_IDS = [
    'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
    'chronic_disease', 'fatigue', 'allergy', 'wheezing',
    'alcohol', 'coughing', 'shortness_of_breath',
    'swallowing_difficulty', 'chest_pain',
];

// ── Form Submission ─────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Validate required fields
    const gender = document.getElementById('gender');
    const age = document.getElementById('age');
    let valid = true;

    // Clear previous errors
    document.querySelectorAll('.field-error').forEach(el => el.classList.remove('field-error'));

    if (!gender.value) {
        gender.closest('.field').classList.add('field-error');
        valid = false;
    }

    const ageVal = parseInt(age.value, 10);
    if (!age.value || isNaN(ageVal) || ageVal < 18 || ageVal > 100) {
        age.closest('.field').classList.add('field-error');
        valid = false;
    }

    if (!valid) {
        // Scroll to the first error
        const firstError = document.querySelector('.field-error');
        if (firstError) firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
    }

    // Build request body
    const payload = {
        model: document.getElementById('modelSelect').value,
        gender: parseInt(gender.value, 10),
        age: ageVal,
    };

    // Add all toggle values (checked = 1, unchecked = 0)
    TOGGLE_IDS.forEach(id => {
        payload[id] = document.getElementById(id).checked ? 1 : 0;
    });

    // Show loading state
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || `Server returned ${res.status}`);
        }

        const data = await res.json();
        showResult(data);

    } catch (err) {
        showError(err.message);
    } finally {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

// ── Display Prediction Result ───────────────────────────────────
function showResult(data) {
    const isHigh = data.prediction === 1;

    resultsDiv.innerHTML = `
        <div class="result-card ${isHigh ? 'result-high' : 'result-low'}">
            <div class="result-icon">${isHigh ? '⚠️' : '✅'}</div>
            <h3 class="result-heading">${isHigh ? 'High Risk Detected' : 'Low Risk Detected'}</h3>
            <p class="result-text">
                ${isHigh
                    ? 'The model predicts a <strong>high risk</strong> of lung cancer based on your responses. Please consult a healthcare professional for proper evaluation and screening.'
                    : 'The model predicts a <strong>low risk</strong> of lung cancer based on your responses. Continue maintaining a healthy lifestyle and schedule regular check-ups.'
                }
            </p>
            <p class="result-model">Model: ${data.model}</p>
            <div class="result-disclaimer">
                <strong>⚠ Disclaimer</strong>
                This is not a medical diagnosis. This tool is for educational purposes only.
                Please consult a qualified healthcare professional for proper medical evaluation.
            </div>
            <button type="button" class="btn btn-retry" onclick="resetAndRetry()">Take Another Assessment</button>
        </div>
    `;

    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Display Error ───────────────────────────────────────────────
function showError(message) {
    let helpText = 'Check that the Flask API is running.';
    if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
        helpText = 'The API server is not reachable. Start it with: <code>python api.py</code>';
    }

    resultsDiv.innerHTML = `
        <div class="result-card result-error">
            <div class="result-icon">🔌</div>
            <h3 class="result-heading">Connection Error</h3>
            <p class="result-text">
                Could not reach the prediction server.<br>
                <span style="font-size:0.85rem; opacity:0.7;">${helpText}</span>
            </p>
            <p class="result-model" style="color: #F87171;">${message}</p>
            <button type="button" class="btn btn-retry" onclick="resetAndRetry()">Try Again</button>
        </div>
    `;

    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Reset Form & Results ────────────────────────────────────────
function resetAndRetry() {
    form.reset();
    resultsDiv.classList.add('hidden');
    resultsDiv.innerHTML = '';
    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
}

// ── Smooth Scroll for Nav Links ─────────────────────────────────
document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
});

// ── Clear validation errors on input ────────────────────────────
document.querySelectorAll('.field input, .field select').forEach(el => {
    el.addEventListener('input', () => {
        el.closest('.field')?.classList.remove('field-error');
    });
    el.addEventListener('change', () => {
        el.closest('.field')?.classList.remove('field-error');
    });
});
