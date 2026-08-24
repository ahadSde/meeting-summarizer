const form = document.querySelector('#upload-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');

const escapeHtml = (value) => String(value ?? 'Unassigned').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
const list = (items, render) => items?.length ? `<ul class="mt-3 space-y-2 text-slate-300">${items.map(render).join('')}</ul>` : '<p class="mt-3 text-slate-500">None identified.</p>';

function render(meeting) {
  const s = meeting.summary;
  results.innerHTML = `
    <article class="rounded-2xl border border-slate-700 bg-slate-900 p-6"><h2 class="text-xl font-semibold">Overview</h2><p class="mt-3 leading-7 text-slate-300">${escapeHtml(s.overview)}</p>${list(s.key_points, p => `<li>• ${escapeHtml(p)}</li>`)}</article>
    <div class="grid gap-6 md:grid-cols-2">
      <article class="rounded-2xl border border-slate-700 bg-slate-900 p-6"><h2 class="text-xl font-semibold">Decisions</h2>${list(s.decisions, d => `<li><b class="text-cyan-300">${escapeHtml(d.timestamp)}</b> ${escapeHtml(d.text)}</li>`)}</article>
      <article class="rounded-2xl border border-slate-700 bg-slate-900 p-6"><h2 class="text-xl font-semibold">Action items</h2>${list(s.action_items, a => `<li><b>${escapeHtml(a.task)}</b><br><span class="text-sm text-slate-400">${escapeHtml(a.owner)} · ${escapeHtml(a.deadline)} · ${escapeHtml(a.timestamp)}</span></li>`)}</article>
    </div>
    <article class="rounded-2xl border border-slate-700 bg-slate-900 p-6"><h2 class="text-xl font-semibold">Transcript</h2><pre class="mt-4 max-h-96 overflow-auto whitespace-pre-wrap font-sans leading-7 text-slate-300">${escapeHtml(meeting.transcript)}</pre></article>`;
  results.classList.remove('hidden');
}

async function poll(id) {
  const response = await fetch(`/api/meetings/${id}`);
  const meeting = await response.json();
  status.textContent = meeting.status === 'failed' ? meeting.error_message : `${meeting.stage}…`;
  if (meeting.status === 'completed') { status.textContent = `Completed in ${meeting.processing_seconds}s`; render(meeting); return true; }
  if (meeting.status === 'failed') return true;
  await new Promise((resolve) => setTimeout(resolve, 1500));
  return poll(id);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button');
  const fileInput = form.querySelector('input[type="file"]');
  const formData = new FormData(form);   // capture the file BEFORE disabling inputs
  button.disabled = true; fileInput.disabled = true; results.classList.add('hidden'); status.classList.remove('hidden'); status.textContent = 'Uploading audio…';
  try {
    const response = await fetch('/api/meetings', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Upload failed.');
    await poll(payload.id);   // wait for processing to actually finish (or fail)
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; fileInput.disabled = false; }
});