/**
 * Rosa-sinensis Leaf Health Diagnostic - Pastel Minimalist Frontend Logic
 * Includes Dual Upload & Direct Camera "Click Picture" Capture
 */

// Application State
const state = {
  selectedFile: null,
  selectedSampleId: null,
  previewDataUrl: null,
  lastResult: null,
  isScanning: false,
  cameraStream: null,
  capturedBlob: null,
  capturedDataUrl: null,
};

// DOM References
const DOM = {
  uploadSection: document.getElementById('uploadSection'),
  scanningSection: document.getElementById('scanningSection'),
  reportSection: document.getElementById('reportSection'),

  // Upload & Camera elements
  tabUpload: document.getElementById('tabUpload'),
  btnOpenCamera: document.getElementById('btnOpenCamera'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  mobileCameraInput: document.getElementById('mobileCameraInput'),
  previewBox: document.getElementById('previewBox'),
  previewImg: document.getElementById('previewImg'),
  previewName: document.getElementById('previewName'),
  previewSize: document.getElementById('previewSize'),
  btnRemove: document.getElementById('btnRemove'),
  btnScan: document.getElementById('btnScan'),
  sampleButtons: document.querySelectorAll('.btn-sample'),

  // Camera Modal elements
  cameraModal: document.getElementById('cameraModal'),
  cameraVideo: document.getElementById('cameraVideo'),
  cameraSnapshotPreview: document.getElementById('cameraSnapshotPreview'),
  btnCloseCamera: document.getElementById('btnCloseCamera'),
  btnShutter: document.getElementById('btnShutter'),
  liveCaptureControls: document.getElementById('liveCaptureControls'),
  confirmCaptureControls: document.getElementById('confirmCaptureControls'),
  btnRetakePhoto: document.getElementById('btnRetakePhoto'),
  btnUseCapturedPhoto: document.getElementById('btnUseCapturedPhoto'),

  // Scanning animation elements
  scanningThumb: document.getElementById('scanningThumb'),
  scanStatusTitle: document.getElementById('scanStatusTitle'),
  scanStatusSub: document.getElementById('scanStatusSub'),

  // Report elements
  statusCard: document.getElementById('statusCard'),
  statusIcon: document.getElementById('statusIcon'),
  statusTitle: document.getElementById('statusTitle'),
  statusBadge: document.getElementById('statusBadge'),
  findingText: document.getElementById('findingText'),
  metricsGrid: document.getElementById('metricsGrid'),
  imgOriginal: document.getElementById('imgOriginal'),
  imgVeins: document.getElementById('imgVeins'),
  imgChlorosis: document.getElementById('imgChlorosis'),

  // 12 stages drawer
  btnToggleStages: document.getElementById('btnToggleStages'),
  stagesArrow: document.getElementById('stagesArrow'),
  stagesPanel: document.getElementById('stagesPanel'),

  // Report Actions
  btnDownload: document.getElementById('btnDownload'),
  btnPrint: document.getElementById('btnPrint'),
  btnScanAnother: document.getElementById('btnScanAnother'),

  // Lightbox Modal
  modalOverlay: document.getElementById('modalOverlay'),
  modalImg: document.getElementById('modalImg'),
  modalClose: document.getElementById('modalClose'),
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
});

function setupEventListeners() {
  // Tab Upload click
  DOM.tabUpload.addEventListener('click', () => {
    DOM.tabUpload.classList.add('active');
    DOM.btnOpenCamera.classList.remove('active');
    DOM.fileInput.click();
  });

  // Dropzone click & file picking
  DOM.dropzone.addEventListener('click', () => DOM.fileInput.click());

  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  DOM.mobileCameraInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(name => {
    DOM.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      DOM.dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    DOM.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      DOM.dropzone.classList.remove('dragover');
    });
  });

  DOM.dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Remove preview
  DOM.btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  // Camera Open
  DOM.btnOpenCamera.addEventListener('click', openCameraViewfinder);
  DOM.btnCloseCamera.addEventListener('click', closeCameraViewfinder);

  // Shutter: Click Picture!
  DOM.btnShutter.addEventListener('click', captureCameraPhoto);
  DOM.btnRetakePhoto.addEventListener('click', retakeCameraPhoto);
  DOM.btnUseCapturedPhoto.addEventListener('click', useCapturedPhoto);

  // Quick sample buttons
  DOM.sampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleId = btn.dataset.sample;
      selectSample(sampleId);
    });
  });

  // Scan button
  DOM.btnScan.addEventListener('click', startScanningFlow);

  // Toggle 12 stages
  DOM.btnToggleStages.addEventListener('click', () => {
    const isOpen = DOM.stagesPanel.classList.toggle('open');
    DOM.stagesArrow.textContent = isOpen ? '▲' : '▼';
  });

  // Action buttons
  DOM.btnDownload.addEventListener('click', downloadReport);
  DOM.btnPrint.addEventListener('click', () => window.print());
  DOM.btnScanAnother.addEventListener('click', resetToUpload);

  // Lightbox Modal Close
  DOM.modalClose.addEventListener('click', closeModal);
  DOM.modalOverlay.addEventListener('click', (e) => {
    if (e.target === DOM.modalOverlay) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal();
      closeCameraViewfinder();
    }
  });
}

// ============================================================================
// Camera / "Click Picture" Feature
// ============================================================================
async function openCameraViewfinder() {
  DOM.btnOpenCamera.classList.add('active');
  DOM.tabUpload.classList.remove('active');

  // Check camera support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // Fallback to mobile native capture
    DOM.mobileCameraInput.click();
    return;
  }

  try {
    // Request environment (rear) camera on phones or webcam on desktops
    const constraints = {
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    };

    state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    DOM.cameraVideo.srcObject = state.cameraStream;
    DOM.cameraVideo.style.display = 'block';
    DOM.cameraSnapshotPreview.style.display = 'none';

    DOM.liveCaptureControls.style.display = 'block';
    DOM.confirmCaptureControls.style.display = 'none';

    DOM.cameraModal.classList.add('active');
  } catch (err) {
    console.warn('Camera stream error, using device capture dialog fallback:', err);
    DOM.mobileCameraInput.click();
  }
}

function captureCameraPhoto() {
  if (!state.cameraStream) return;

  const video = DOM.cameraVideo;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  state.capturedDataUrl = canvas.toDataURL('image/jpeg', 0.92);

  canvas.toBlob((blob) => {
    state.capturedBlob = blob;
  }, 'image/jpeg', 0.92);

  // Show captured snapshot in viewfinder
  DOM.cameraSnapshotPreview.src = state.capturedDataUrl;
  DOM.cameraSnapshotPreview.style.display = 'block';
  DOM.cameraVideo.style.display = 'none';

  // Toggle buttons
  DOM.liveCaptureControls.style.display = 'none';
  DOM.confirmCaptureControls.style.display = 'flex';
}

function retakeCameraPhoto() {
  DOM.cameraSnapshotPreview.style.display = 'none';
  DOM.cameraVideo.style.display = 'block';

  DOM.liveCaptureControls.style.display = 'block';
  DOM.confirmCaptureControls.style.display = 'none';
  state.capturedBlob = null;
  state.capturedDataUrl = null;
}

function useCapturedPhoto() {
  if (!state.capturedBlob) return;

  const photoFile = new File([state.capturedBlob], `camera_leaf_${Date.now()}.jpg`, {
    type: 'image/jpeg',
  });

  handleFileSelected(photoFile);
  closeCameraViewfinder();
}

function closeCameraViewfinder() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  DOM.cameraModal.classList.remove('active');
  DOM.btnOpenCamera.classList.remove('active');
  DOM.tabUpload.classList.add('active');
}

// ============================================================================
// File Selection Handler
// ============================================================================
function handleFileSelected(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file (JPEG or PNG).');
    return;
  }

  state.selectedFile = file;
  state.selectedSampleId = null;
  DOM.sampleButtons.forEach((b) => b.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = (e) => {
    state.previewDataUrl = e.target.result;
    DOM.previewImg.src = state.previewDataUrl;
    DOM.previewName.textContent = file.name;
    DOM.previewSize.textContent = `${(file.size / 1024).toFixed(1)} KB • Ready to scan`;
    DOM.previewBox.classList.add('active');
  };
  reader.readAsDataURL(file);
}

// Select a Preset Sample
function selectSample(sampleId) {
  state.selectedSampleId = sampleId;
  state.selectedFile = null;
  DOM.fileInput.value = '';
  DOM.mobileCameraInput.value = '';
  DOM.previewBox.classList.remove('active');

  DOM.sampleButtons.forEach((b) => {
    b.classList.toggle('active', b.dataset.sample === sampleId);
  });

  startScanningFlow();
}

function clearSelectedFile() {
  state.selectedFile = null;
  state.previewDataUrl = null;
  DOM.fileInput.value = '';
  DOM.mobileCameraInput.value = '';
  DOM.previewBox.classList.remove('active');
}

// ============================================================================
// Scanning Flow
// ============================================================================
async function startScanningFlow() {
  if (state.isScanning) return;

  if (!state.selectedFile && !state.selectedSampleId) {
    selectSample('healthy');
    return;
  }

  state.isScanning = true;

  // Transition UI: show scanning card
  DOM.uploadSection.style.display = 'none';
  DOM.reportSection.classList.remove('active');
  DOM.scanningSection.classList.add('active');

  // Set scanning thumbnail
  if (state.previewDataUrl) {
    DOM.scanningThumb.src = state.previewDataUrl;
  } else {
    DOM.scanningThumb.src =
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 24 24' fill='%234a8b64'%3E%3Cpath d='M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75C7 8 17 8 17 8z'/%3E%3C/svg%3E";
  }

  DOM.scanStatusTitle.textContent = 'Scanning leaf veins...';
  DOM.scanStatusSub.textContent = 'Extracting vascular skeleton & chlorosis color index';

  try {
    const formData = new FormData();
    if (state.selectedSampleId) {
      formData.append('demo_sample', state.selectedSampleId);
    } else if (state.selectedFile) {
      formData.append('frontlit_image', state.selectedFile);
    }

    const fetchPromise = fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    // 1.2s delay for satisfying visual scan
    const [response] = await Promise.all([
      fetchPromise,
      new Promise((r) => setTimeout(r, 1200)),
    ]);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis could not be completed.');
    }

    const data = await response.json();
    state.lastResult = data;

    renderReport(data);
  } catch (error) {
    console.error('Scan failed:', error);
    alert(`Could not complete diagnosis: ${error.message}\nPlease make sure the python backend is running.`);
    resetToUpload();
  } finally {
    state.isScanning = false;
  }
}

// ============================================================================
// Render Diagnostic Report
// ============================================================================
function renderReport(data) {
  DOM.scanningSection.classList.remove('active');
  DOM.reportSection.classList.add('active');

  // 1. Status Card & Pastel Tone
  const statusKey = data.status || 'HEALTHY';
  DOM.statusCard.className = `status-card status-${statusKey.toLowerCase()}`;

  const badge = data.status_badge || {};
  DOM.statusTitle.textContent = badge.label || 'Leaf Diagnostic';
  DOM.statusBadge.textContent = getShortBadgeLabel(statusKey);
  DOM.statusIcon.innerHTML = getStatusIcon(statusKey);

  // 2. Botanical Finding
  DOM.findingText.textContent = data.explanation || 'Normal physiological baseline observed.';

  // 3. Metrics Grid (5 Clean Cards)
  renderMetrics(data.metrics || []);

  // 4. Visual Verification (3 Cards)
  const g = data.gallery || {};
  DOM.imgOriginal.src = g.original_frontlit || '';
  DOM.imgVeins.src = g.vein_skeleton_overlay || '';
  DOM.imgChlorosis.src = g.chlorosis_overlay || '';

  // 5. Populate 12 Stages in accordion
  renderStages(data.stages || []);
}

function getShortBadgeLabel(status) {
  switch (status) {
    case 'HEALTHY':
      return 'Optimal Health';
    case 'MILD':
      return 'Attention (Mild)';
    case 'MODERATE':
      return 'Moderate Stress';
    case 'SEVERE':
      return 'Critical';
    case 'REJECTED':
      return 'Not Hibiscus';
    default:
      return 'Evaluated';
  }
}

function getStatusIcon(status) {
  switch (status) {
    case 'HEALTHY':
      return `<svg viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>`;
    case 'MILD':
      return `<svg viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    case 'MODERATE':
      return `<svg viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    case 'SEVERE':
      return `<svg viewBox="0 0 24 24" fill="none" stroke="#e11d48" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    default:
      return `<svg viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>`;
  }
}

function renderMetrics(metrics) {
  DOM.metricsGrid.innerHTML = '';
  metrics.forEach((m) => {
    const card = document.createElement('div');
    card.className = 'metric-pill';

    const statusType = m.status_type || 'optimal';

    card.innerHTML = `
      <div class="metric-pill-label">${m.name}</div>
      <div class="metric-pill-val">${m.value} <span>${m.unit}</span></div>
      <div class="metric-pill-status ${statusType}">${m.status}</div>
    `;
    DOM.metricsGrid.appendChild(card);
  });
}

function renderStages(stages) {
  DOM.stagesPanel.innerHTML = '';
  stages.forEach((st) => {
    const item = document.createElement('div');
    item.className = 'stage-mini-card';
    item.innerHTML = `
      <img src="${st.image}" alt="${st.name}" onclick="openModal('${st.image}')">
      <div class="stage-mini-name">#${st.stage} ${st.name}</div>
    `;
    DOM.stagesPanel.appendChild(item);
  });
}

function resetToUpload() {
  DOM.scanningSection.classList.remove('active');
  DOM.reportSection.classList.remove('active');
  DOM.uploadSection.style.display = 'block';

  clearSelectedFile();
  DOM.sampleButtons.forEach((b) => b.classList.remove('active'));
  state.selectedSampleId = null;
  state.lastResult = null;
}

// Lightbox Modal
function openModal(src) {
  if (!src) return;
  DOM.modalImg.src = src;
  DOM.modalOverlay.classList.add('active');
}

function closeModal() {
  DOM.modalOverlay.classList.remove('active');
  DOM.modalImg.src = '';
}

// Download Full Diagnostic Report
function downloadReport() {
  if (!state.lastResult) return;

  const d = state.lastResult;
  const now = new Date().toLocaleString();

  let text = `=======================================================
ROSA-SINENSIS LEAF HEALTH REPORT
Date: ${now}
=======================================================

DIAGNOSIS: ${d.status_badge?.label || d.status}
ETIOLOGY : ${d.etiology}

BOTANICAL SUMMARY:
${d.explanation}

QUANTITATIVE MEASUREMENTS:
`;

  (d.metrics || []).forEach((m) => {
    text += `- ${m.name}: ${m.value} ${m.unit} (${m.status}, Healthy range: ${m.healthy_min}-${m.healthy_max})\n`;
  });

  if (d.recommendations && d.recommendations.length > 0) {
    text += `\nCARE RECOMMENDATIONS:\n`;
    d.recommendations.forEach((r, idx) => {
      text += `${idx + 1}. ${r}\n`;
    });
  }

  text += `\n=======================================================\n`;

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `Hibiscus_Report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
