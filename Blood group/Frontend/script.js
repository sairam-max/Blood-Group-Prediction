const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    preview.classList.remove("hidden");
    previewImg.src = URL.createObjectURL(file);
  }
});

async function uploadImage() {
  const file = fileInput.files[0];
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  if (!file) {
    alert("Please upload a fingerprint image first!");
    return;
  }

  loading.classList.remove("hidden");
  result.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    document.getElementById("bloodType").textContent = data.predicted_blood_type;
    const confidencePercent = (data.confidence * 100).toFixed(2);
    document.getElementById("confidence").textContent = confidencePercent;
    document.getElementById("bar").style.width = confidencePercent + "%";

    loading.classList.add("hidden");
    result.classList.remove("hidden");
  } catch (error) {
    console.error("Error:", error);
    alert("Prediction failed. Please check your backend server.");
    loading.classList.add("hidden");
  }
}
