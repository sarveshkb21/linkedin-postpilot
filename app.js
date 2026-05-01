'use strict';

// ── Storage helpers ────────────────────────────────────────────
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function save(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function todayKey() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

// ── Default data ───────────────────────────────────────────────
const DEFAULT_LINKS = [
  { id: uid(), url: 'https://github.com',        label: 'GitHub' },
  { id: uid(), url: 'https://mail.google.com',   label: 'Gmail' },
  { id: uid(), url: 'https://calendar.google.com', label: 'Calendar' },
];

// ── State ──────────────────────────────────────────────────────
let tasks    = load('tasks', []);
let links    = load('links', DEFAULT_LINKS);
let userName = load('userName', 'Sarvesh');

// ── DOM refs ───────────────────────────────────────────────────
const clockEl        = document.getElementById('clock');
const dateEl         = document.getElementById('date');
const greetingEl     = document.getElementById('greeting');
const nameDisplay    = document.getElementById('user-name-display');
const nameInput      = document.getElementById('user-name-input');
const editNameBtn    = document.getElementById('edit-name-btn');

const focusInput     = document.getElementById('focus-input');
const focusChars     = document.getElementById('focus-chars');

const taskInput      = document.getElementById('task-input');
const taskAddBtn     = document.getElementById('task-add-btn');
const taskList       = document.getElementById('task-list');
const taskCount      = document.getElementById('task-count');

const linkAddBtn     = document.getElementById('link-add-btn');
const linkForm       = document.getElementById('link-form');
const linkUrl        = document.getElementById('link-url');
const linkLabel      = document.getElementById('link-label');
const linkSaveBtn    = document.getElementById('link-save-btn');
const linkCancelBtn  = document.getElementById('link-cancel-btn');
const linkList       = document.getElementById('link-list');

// ── Clock & greeting ───────────────────────────────────────────
function tickClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  const s = String(now.getSeconds()).padStart(2, '0');
  clockEl.textContent = `${h}:${m}:${s}`;

  const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  dateEl.textContent = `${days[now.getDay()]}, ${months[now.getMonth()]} ${now.getDate()} · ${now.getFullYear()}`;
}

function updateGreeting() {
  const hour = new Date().getHours();
  const part = hour < 12 ? 'morning' : hour < 17 ? 'afternoon' : 'evening';
  greetingEl.textContent = `Good ${part},`;
}

// ── User name ──────────────────────────────────────────────────
function renderName() {
  nameDisplay.textContent = userName;
}

function startEditName() {
  nameInput.value = userName;
  nameDisplay.classList.add('hidden');
  editNameBtn.classList.add('hidden');
  nameInput.classList.remove('hidden');
  nameInput.focus();
  nameInput.select();
}

function commitName() {
  const val = nameInput.value.trim();
  if (val) userName = val;
  save('userName', userName);
  nameInput.classList.add('hidden');
  nameDisplay.classList.remove('hidden');
  editNameBtn.classList.remove('hidden');
  renderName();
  updateGreeting();
}

editNameBtn.addEventListener('click', startEditName);
nameInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') commitName();
  if (e.key === 'Escape') {
    nameInput.classList.add('hidden');
    nameDisplay.classList.remove('hidden');
    editNameBtn.classList.remove('hidden');
  }
});
nameInput.addEventListener('blur', commitName);

// ── Focus of the Day ───────────────────────────────────────────
function focusStorageKey() {
  return `focus_${todayKey()}`;
}

function loadFocus() {
  focusInput.value = load(focusStorageKey(), '');
  updateFocusChars();
}

let focusDebounce = null;
function saveFocus() {
  clearTimeout(focusDebounce);
  focusDebounce = setTimeout(() => save(focusStorageKey(), focusInput.value), 300);
}

function updateFocusChars() {
  const len = focusInput.value.length;
  focusChars.textContent = `${len} / 120`;
  focusChars.classList.toggle('near-limit', len >= 100 && len < 120);
  focusChars.classList.toggle('at-limit', len >= 120);
}

focusInput.addEventListener('input', () => {
  updateFocusChars();
  saveFocus();
});

// ── Tasks ──────────────────────────────────────────────────────
function saveTasks() { save('tasks', tasks); }

function renderTasks() {
  taskList.innerHTML = '';
  const sorted = [...tasks].sort((a, b) => a.done - b.done);
  sorted.forEach(task => {
    const li = document.createElement('li');
    li.className = 'task-item' + (task.done ? ' done' : '');
    li.dataset.id = task.id;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = task.done;
    cb.addEventListener('change', () => toggleTask(task.id));

    const label = document.createElement('span');
    label.className = 'task-label';
    label.textContent = task.text;

    const del = document.createElement('button');
    del.className = 'task-delete';
    del.textContent = '×';
    del.setAttribute('aria-label', 'Delete task');
    del.addEventListener('click', () => deleteTask(task.id));

    li.append(cb, label, del);
    taskList.appendChild(li);
  });

  const done  = tasks.filter(t => t.done).length;
  const total = tasks.length;
  taskCount.textContent = total ? `${done} / ${total}` : '';
  taskCount.style.display = total ? '' : 'none';
}

function addTask(text) {
  if (!text) return;
  tasks.push({ id: uid(), text, done: false });
  saveTasks();
  renderTasks();
}

function toggleTask(id) {
  const t = tasks.find(t => t.id === id);
  if (t) { t.done = !t.done; saveTasks(); renderTasks(); }
}

function deleteTask(id) {
  tasks = tasks.filter(t => t.id !== id);
  saveTasks();
  renderTasks();
}

taskAddBtn.addEventListener('click', () => {
  addTask(taskInput.value.trim());
  taskInput.value = '';
});

taskInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    addTask(taskInput.value.trim());
    taskInput.value = '';
  }
});

// ── Quick Links ────────────────────────────────────────────────
function saveLinks() { save('links', links); }

function faviconUrl(url) {
  try {
    const domain = new URL(url).hostname;
    return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
  } catch {
    return '';
  }
}

function renderLinks() {
  linkList.innerHTML = '';
  links.forEach(link => {
    const li = document.createElement('li');
    li.className = 'link-item';
    li.dataset.id = link.id;

    const a = document.createElement('a');
    a.href = link.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';

    const img = document.createElement('img');
    img.className = 'link-favicon';
    img.src = faviconUrl(link.url);
    img.alt = '';
    img.onerror = () => { img.style.display = 'none'; };

    const labelEl = document.createElement('span');
    labelEl.textContent = link.label || link.url;

    a.append(img, labelEl);

    const del = document.createElement('button');
    del.className = 'link-delete';
    del.textContent = '×';
    del.setAttribute('aria-label', 'Delete link');
    del.addEventListener('click', () => deleteLink(link.id));

    li.append(a, del);
    linkList.appendChild(li);
  });
}

function addLink(url, label) {
  if (!url) return;
  if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
  links.push({ id: uid(), url, label: label || url });
  saveLinks();
  renderLinks();
}

function deleteLink(id) {
  links = links.filter(l => l.id !== id);
  saveLinks();
  renderLinks();
}

function showLinkForm() {
  linkForm.classList.remove('hidden');
  linkUrl.focus();
  linkAddBtn.textContent = '−';
}

function hideLinkForm() {
  linkForm.classList.add('hidden');
  linkUrl.value = '';
  linkLabel.value = '';
  linkAddBtn.textContent = '+';
}

function commitLinkForm() {
  addLink(linkUrl.value.trim(), linkLabel.value.trim());
  hideLinkForm();
}

linkAddBtn.addEventListener('click', () => {
  linkForm.classList.contains('hidden') ? showLinkForm() : hideLinkForm();
});

linkSaveBtn.addEventListener('click', commitLinkForm);
linkCancelBtn.addEventListener('click', hideLinkForm);

linkUrl.addEventListener('keydown', e => {
  if (e.key === 'Enter') { linkLabel.focus(); }
  if (e.key === 'Escape') hideLinkForm();
});
linkLabel.addEventListener('keydown', e => {
  if (e.key === 'Enter') commitLinkForm();
  if (e.key === 'Escape') hideLinkForm();
});

// ── Keyboard shortcuts ─────────────────────────────────────────
document.addEventListener('keydown', e => {
  // Skip when focused on an input/textarea
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  if (e.key === 'n' || e.key === 'N') {
    taskInput.focus();
  }
  if (e.key === 'f' || e.key === 'F') {
    focusInput.focus();
    focusInput.setSelectionRange(focusInput.value.length, focusInput.value.length);
  }
});

// ── Init ───────────────────────────────────────────────────────
tickClock();
setInterval(tickClock, 1000);
setInterval(updateGreeting, 60_000);

updateGreeting();
renderName();
loadFocus();
renderTasks();
renderLinks();
