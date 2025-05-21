// --- Standard Functionality ---

// Hide header on scroll down
let lastScrollPosition = 0;
const header = document.getElementById("header");
const scrollableElement = document.documentElement;

window.addEventListener("scroll", () => {
  const currentScrollPosition = scrollableElement.scrollTop;
  if (header) {
    if (
      currentScrollPosition > lastScrollPosition &&
      currentScrollPosition > header.offsetHeight
    ) {
      header.classList.add("hidden");
    } else {
      header.classList.remove("hidden");
    }
  }
  lastScrollPosition = currentScrollPosition <= 0 ? 0 : currentScrollPosition;
});

// Tab functionality (Modified to save active tab to localStorage)
function openTab(tabName) {
  const tabsContainer = document.getElementById("tabs-container");
  const trainTabContent = document.getElementById("train");
  const testTabContent = document.getElementById("test");
  const trainBtn = document.getElementById("train-btn");
  const testBtn = document.getElementById("test-btn");

  // Check if elements exist before removing classes
  if (trainTabContent) trainTabContent.classList.remove("active");
  if (testTabContent) testTabContent.classList.remove("active");
  if (trainBtn) trainBtn.classList.remove("active");
  if (testBtn) testBtn.classList.remove("active");
  if (tabsContainer) tabsContainer.classList.remove("train-mode", "test-mode");

  // Open the selected tab and save to localStorage
  if (tabName === "train" && trainTabContent) {
    trainTabContent.classList.add("active");
    if (trainBtn) trainBtn.classList.add("active");
    if (tabsContainer) tabsContainer.classList.add("train-mode");
    localStorage.setItem("activeTab", "train"); // Save the active tab name
  } else if (tabName === "test" && testTabContent) {
    testTabContent.classList.add("active");
    if (testBtn) testBtn.classList.add("active");
    if (tabsContainer) tabsContainer.classList.add("test-mode");
    localStorage.setItem("activeTab", "test"); // Save the active tab name
  } else {
    // Optional: Handle case where requested tab doesn't exist or is invalid
    console.warn("Attempted to open invalid tab:", tabName);
    // If you want to default to 'train' in this case:
    // if (trainTabContent) openTab('train'); // Recursive call, be careful
  }
}

// Function to update file label (only sets name if a file is selected, keeps HTML default otherwise)
function updateFileLabel(inputId, labelId) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);

  if (!input || !label) {
    console.warn(
      `updateFileLabel: Input with ID '${inputId}' or label with ID '${labelId}' not found.`
    );
    return;
  }

  if (input.files && input.files.length > 0) {
    label.textContent = input.files[0].name;
  }
  // Do NOT set text if no files. The default HTML text will remain visible.
}

// --- Hyperparameter and Persistence Logic ---
document.addEventListener("DOMContentLoaded", function () {
  // --- Get DOM References ---
  const cnnParamsDiv = document.getElementById("cnn-hyperparams");
  const svmParamsDiv = document.getElementById("svm-hyperparams"); // Corrected to svm-
  const vitParamsDiv = document.getElementById("vit-hyperparams");
  const modelSelect = document.getElementById("train-model-select");

  // --- SVM specific elements for conditional display ---
  const svmKernelSelect = document.getElementById("svm-kernel"); // Corrected to svm-
  const svmPolyDegreeDiv = document.getElementById("svm-poly-degree"); // Corrected to svm-
  const svmCoef0Div = document.getElementById("svm-coef0"); // Corrected to svm-

  // --- Function to toggle hyperparameter divs based on model selection ---
  window.toggleHyperparams = function () {
    if (!modelSelect) {
      console.error("Model select dropdown not found!");
      return;
    }
    const selectedModelDivId = modelSelect.value;

    if (cnnParamsDiv) cnnParamsDiv.style.display = "none";
    if (svmParamsDiv) svmParamsDiv.style.display = "none";
    if (vitParamsDiv) vitParamsDiv.style.display = "none";

    const selectedDiv = document.getElementById(selectedModelDivId);
    if (selectedDiv) {
      selectedDiv.style.display = "flex";

      if (selectedModelDivId === "svm-hyperparams") {
        setTimeout(function () {
          const svmKernelSelectInsideToggle =
            document.getElementById("svm-kernel");
          const svmPolyDegreeDivInsideToggle =
            document.getElementById("svm-poly-degree");
          const svmCoef0DivInsideToggle = document.getElementById("svm-coef0");

          if (
            svmKernelSelectInsideToggle &&
            svmPolyDegreeDivInsideToggle &&
            svmCoef0DivInsideToggle
          ) {
            toggleSvmKernelParams();
          } else {
            console.warn(
              "SVM kernel or child divs not found inside toggleHyperparams for SVM specific toggle."
            );
          }
        }, 0);
      }
    }
  };

  // --- SVM Specific Toggle Function ---
  function toggleSvmKernelParams() {
    if (svmKernelSelect && svmPolyDegreeDiv && svmCoef0Div) {
      const kernel = svmKernelSelect.value;
      svmPolyDegreeDiv.style.display = kernel === "poly" ? "flex" : "none";
      svmCoef0Div.style.display =
        kernel === "poly" || kernel === "sigmoid" ? "flex" : "none";
    } else {
      console.warn(
        "SVM kernel or child divs not found for toggleSvmKernelParams."
      );
    }
  }

  // --- Function to save ALL hyperparameters to localStorage ---
  function saveAllHyperparameters() {
    if (!modelSelect) {
      console.error("Model select dropdown not found for saving!");
      return;
    }
    const selectedModelDivId = modelSelect.value;

    const allHyperparameters = {
      model: selectedModelDivId,
    };

    [cnnParamsDiv, svmParamsDiv, vitParamsDiv].forEach((paramsDiv) => {
      if (paramsDiv) {
        paramsDiv.querySelectorAll("input, select").forEach((input) => {
          if (input.name) {
            allHyperparameters[input.name] = input.value;
          }
        });
      }
    });

    localStorage.setItem(
      "lastHyperparameters",
      JSON.stringify(allHyperparameters)
    );
  }

  // --- Function to load ALL hyperparameters from localStorage and apply to form ---
  function loadAllHyperparameters() {
    const savedData = localStorage.getItem("lastHyperparameters");

    if (savedData) {
      try {
        const allHyperparameters = JSON.parse(savedData);

        if (allHyperparameters.model && modelSelect) {
          const optionExists = Array.from(modelSelect.options).some(
            (option) => option.value === allHyperparameters.model
          );
          if (optionExists) {
            modelSelect.value = allHyperparameters.model;
          } else {
            console.warn(
              "Saved model value not found in select options. Loading default."
            );
            if (modelSelect.options.length > 0) modelSelect.selectedIndex = 0;
          }
        } else if (modelSelect && modelSelect.options.length > 0) {
          modelSelect.selectedIndex = 0;
        } else if (!modelSelect) {
          console.error("Model select dropdown not found for loading!");
          return;
        } else {
          console.warn("Model select dropdown found but has no options.");
          return;
        }

        [cnnParamsDiv, svmParamsDiv, vitParamsDiv].forEach((paramsDiv) => {
          if (paramsDiv) {
            paramsDiv.querySelectorAll("input, select").forEach((input) => {
              if (input.name && allHyperparameters.hasOwnProperty(input.name)) {
                if (input.type === "checkbox") {
                  input.checked = allHyperparameters[input.name] === "true";
                } else {
                  input.value = allHyperparameters[input.name];
                }
              }
            });
          }
        });

        setTimeout(function () {
          toggleHyperparams();
        }, 0);
      } catch (e) {
        console.error(
          "Error loading or parsing hyperparameters from localStorage:",
          e
        );
        localStorage.removeItem("lastHyperparameters");
        if (modelSelect && modelSelect.options.length > 0)
          modelSelect.selectedIndex = 0;
        setTimeout(toggleHyperparams, 0);
      }
    } else {
      if (modelSelect && modelSelect.options.length > 0)
        modelSelect.selectedIndex = 0;
      setTimeout(toggleHyperparams, 0);
    }
  }

  // --- Add Event Listeners to save ALL parameters on change ---

  if (modelSelect) {
    modelSelect.addEventListener("change", saveAllHyperparameters);
  } else {
    console.error(
      "Model select dropdown not found for adding event listeners!"
    );
  }

  [cnnParamsDiv, svmParamsDiv, vitParamsDiv].forEach((paramsDiv) => {
    if (paramsDiv) {
      paramsDiv.addEventListener("change", function (event) {
        if (
          event.target.tagName === "INPUT" ||
          event.target.tagName === "SELECT"
        ) {
          saveAllHyperparameters();
        }
      });
    }
  });

  if (svmKernelSelect) {
    svmKernelSelect.addEventListener("change", function () {
      toggleSvmKernelParams();
    });
  }

  // --- Initial Page Setup ---
  // Check localStorage for a saved active tab
  const savedTab = localStorage.getItem("activeTab");

  if (savedTab) {
    // Open the saved tab if one exists
    openTab(savedTab);
  } else {
    // If no saved tab, open the default tab (e.g., 'train')
    openTab("train");
  }

  // Update initial file labels (these calls rely on the 'onchange' in HTML to trigger updates)
  // They are here mostly to ensure the function exists if HTML triggers it early.
  updateFileLabel("train-zip-input", "train-zip-label");
  updateFileLabel("test-file-input", "test-file-label");

  // Load saved hyperparameters and set initial form state
  loadAllHyperparameters();

  window.addEventListener("beforeunload", function () {
    if (
      window.eventSource &&
      window.eventSource.readyState !== EventSource.CLOSED
    ) {
      window.eventSource.close();
      console.log("SSE connection closed on unload.");
    }
  });
}); // End DOMContentLoaded listener
///////

const ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif", "ppm"];

// Function to check if a file has an allowed image extension
function isAllowedImageFile(file) {
  if (!file.name) return false;
  const parts = file.name.split(".");
  if (parts.length === 1) return false; // No extension
  const ext = parts.pop().toLowerCase();
  return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
}
function handleFile(files) {
  // just update preview, no submit here
  if (files.length) {
    showPreview(files[0]);
  }
}

function handleDrop(e) {
  e.preventDefault();
  const dt = e.dataTransfer;
  if (dt.files.length) {
    const input = document.getElementById("test-file-input");

    // Create a DataTransfer to set input.files properly (for browsers that support it)
    const dataTransfer = new DataTransfer();
    for (let file of dt.files) {
      dataTransfer.items.add(file);
    }
    input.files = dataTransfer.files;

    showPreview(dt.files[0]);
    document.getElementById("upload-form").submit();
  }
}

document
  .getElementById("test-file-input")
  .addEventListener("change", function () {
    if (this.files.length) {
      showPreview(this.files[0]);
      this.form.submit();
    }
  });

function showPreview(file) {
  const img = document.getElementById("test-image");
  const dropText = document.getElementById("drop-text");
  const reader = new FileReader();

  reader.onload = function (e) {
    img.src = e.target.result;
    img.style.display = "block";
    dropText.style.display = "none";
  };

  reader.readAsDataURL(file);
}

const select = document.getElementById("test-model-select");

// Load saved selection on page load
window.onload = () => {
  const saved = localStorage.getItem("selectedModel");
  if (saved) select.value = saved;
};

// Save selection on change
select.addEventListener("change", () => {
  localStorage.setItem("selectedModel", select.value);
});

document.querySelectorAll("input[type=number]").forEach((input) => {
  input.addEventListener("input", () => {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    let val = parseFloat(input.value);

    if (!isNaN(min) && val < min) val = min;
    if (!isNaN(max) && val > max) val = max;

    input.value = val;
  });
});
