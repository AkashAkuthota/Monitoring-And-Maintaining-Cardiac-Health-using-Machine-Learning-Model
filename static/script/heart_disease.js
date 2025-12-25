// Bubble range creation
const allRanges = document.querySelectorAll(".range-wrap");
allRanges.forEach(wrap => {
    const range = wrap.querySelector(".range");
    const bubble = wrap.querySelector(".bubble");

    range.addEventListener("input", () => {
        setBubble(range, bubble);
    });
    setBubble(range, bubble);
});

function setBubble(range, bubble) {
    const val = range.value;
    const min = range.min || 0;
    const max = range.max || 100;
    const newVal = Number(((val - min) * 100) / (max - min));
    bubble.innerHTML = val;
    bubble.style.left = `calc(${newVal}% + (${8 - newVal * 0.15}px))`;
}

// Page switching
function replace(hide, show) {
    document.getElementById(hide).style.display = "none";
    document.getElementById(show).style.display = "block";
}

// ✅ FINAL SUBMIT FIX
function submitForm(h) {
    const form = document.getElementById("form-id");
    form.action = "/heart-disease-predictor";
    form.method = "POST";
    document.getElementById(h).style.display = "none";
    form.submit();
}
