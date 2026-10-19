const $ = (id) => document.getElementById(id);

let username = null;
let state = { films: {}, ranked: [], skipped: [] };
let queue = [];
let candidate = null;
let lo = 0;
let hi = 0;
let comparisons = 0;
let totalToPlace = 0;
let placedThisSession = 0;
let undoStack = [];

const views = ["view-setup", "view-compare", "view-done", "view-list"];

function show(viewId) {
  views.forEach((v) => $(v).classList.toggle("hidden", v !== viewId));
  $("nav-rank").classList.toggle("active", viewId !== "view-list");
  $("nav-list").classList.toggle("active", viewId === "view-list");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

function saveState() {
  fetch(`/api/state?username=${encodeURIComponent(username)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  }).catch(() => {});
}

const canonical = (url) => (url || "").replace(/(\/film\/[^/]+)\/\d+\/?$/, "$1/");
const titleYear = (f) => `${(f.title || "").toLowerCase()}|${f.year || ""}`;

function titleYearIndex() {
  const index = {};
  for (const f of Object.values(state.films)) index[titleYear(f)] = f.key;
  return index;
}

function remapKey(oldKey, newKey) {
  state.ranked = state.ranked.map((k) => (k === oldKey ? newKey : k));
  state.skipped = state.skipped.map((k) => (k === oldKey ? newKey : k));
}

function mergeFeedFilm(film, index) {
  const existingKey = state.films[film.key] ? film.key : index[titleYear(film)];
  if (existingKey) {
    const old = state.films[existingKey];
    state.films[film.key] = {
      ...old,
      ...film,
      watched: film.watched || old.watched,
      poster: film.poster || old.poster,
      rating: film.rating ?? old.rating,
    };
    if (existingKey !== film.key) {
      delete state.films[existingKey];
      remapKey(existingKey, film.key);
    }
  } else {
    state.films[film.key] = film;
  }
  index[titleYear(film)] = film.key;
}

function rebuildQueue() {
  const placed = new Set([...state.ranked, ...state.skipped]);
  queue = Object.values(state.films)
    .filter((f) => !placed.has(f.key))
    .sort((a, b) => (b.watched || "").localeCompare(a.watched || ""))
    .map((f) => f.key);
  totalToPlace = placedThisSession + queue.length;
}

let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 5000);
}

const posterQueue = [];
const posterQueued = new Set();
let posterWorking = false;

function queuePoster(film, imgEl, priority) {
  if (film.poster || !/^https:\/\/boxd\.it\//.test(film.key) || posterQueued.has(film.key)) return;
  posterQueued.add(film.key);
  posterQueue[priority ? "unshift" : "push"]([film.key, imgEl]);
  pumpPosters();
}

async function pumpPosters() {
  if (posterWorking) return;
  posterWorking = true;
  while (posterQueue.length) {
    const [key, img] = posterQueue.shift();
    const film = state.films[key];
    posterQueued.delete(key);
    if (!film || film.poster) continue;
    try {
      const res = await api(`/api/poster?uri=${encodeURIComponent(key)}`);
      if (res.poster) film.poster = res.poster;
      if (res.filmUrl) film.url = res.filmUrl;
      saveState();
      if (img && img.dataset.key === key && res.poster) {
        img.src = img.id === "poster-candidate" || img.id === "poster-pivot"
          ? comparisonPoster(res.poster) : res.poster;
        img.hidden = false;
      }
    } catch {
      // leave poster missing; placeholder background shows instead
    }
  }
  posterWorking = false;
}

async function importHistory() {
  toast("Looking for your export zip…");
  try {
    const res = await api(`/api/import?username=${encodeURIComponent(username)}`);
    const index = titleYearIndex();
    let added = 0;
    for (const film of res.films) {
      const existingKey = state.films[film.key] ? film.key : index[titleYear(film)];
      if (existingKey) {
        const old = state.films[existingKey];
        old.watched = old.watched || film.watched;
        old.rating = old.rating ?? film.rating;
      } else {
        state.films[film.key] = film;
        index[titleYear(film)] = film.key;
        added++;
      }
    }
    rebuildQueue();
    saveState();
    toast(`Imported ${res.films.length} films from ${res.source} — ${added} new to rank.`);
    nextCandidate();
  } catch (err) {
    toast(err.message);
  }
}

function migrate(saved) {
  const films = {};
  for (const [key, film] of Object.entries(saved.films || {})) {
    const canon = canonical(film.url || key);
    if (!films[canon]) films[canon] = { ...film, key: canon, url: canon };
  }
  const seen = new Set();
  const remap = (keys) =>
    (keys || [])
      .map(canonical)
      .filter((k) => films[k] && !seen.has(k) && seen.add(k));
  return { films, ranked: remap(saved.ranked), skipped: remap(saved.skipped) };
}

async function init(name) {
  username = name;
  localStorage.setItem("lbx-username", username);
  $("whoami").textContent = `@${username}`;
  $("nav").classList.remove("hidden");

  const [saved, feed] = await Promise.all([
    api(`/api/state?username=${encodeURIComponent(username)}`),
    api(`/api/films?username=${encodeURIComponent(username)}`),
  ]);
  state = migrate(saved);
  const index = titleYearIndex();
  for (const film of feed.films) mergeFeedFilm(film, index);

  placedThisSession = 0;
  undoStack = [];
  rebuildQueue();
  saveState();
  nextCandidate();
}

function nextCandidate() {
  candidate = queue.shift() || null;
  comparisons = 0;
  if (!candidate) {
    show(state.ranked.length ? "view-done" : "view-setup");
    if (state.ranked.length) renderList();
    return;
  }
  if (state.ranked.length === 0) {
    state.ranked.push(candidate);
    placedThisSession++;
    saveState();
    nextCandidate();
    return;
  }
  lo = 0;
  hi = state.ranked.length;
  renderComparison();
}

function renderComparison() {
  const cand = state.films[candidate];
  const mid = Math.floor((lo + hi) / 2);
  const pivot = state.films[state.ranked[mid]];

  show("view-compare");
  $("compare-question").textContent = "Which film did you prefer?";
  $("poster-candidate").src = comparisonPoster(cand.poster);
  $("poster-candidate").alt = "";
  $("poster-candidate").hidden = !cand.poster;
  $("poster-candidate").dataset.key = cand.key;
  $("poster-pivot").dataset.key = pivot.key;
  queuePoster(cand, $("poster-candidate"), true);
  queuePoster(pivot, $("poster-pivot"), true);
  $("title-candidate").textContent = cand.title;
  $("year-candidate").textContent = cand.year;
  $("poster-pivot").src = comparisonPoster(pivot.poster);
  $("poster-pivot").alt = "";
  $("poster-pivot").hidden = !pivot.poster;
  $("title-pivot").textContent = pivot.title;
  $("year-pivot").textContent = pivot.year;
  $("pivot-rank").textContent = `ranked #${mid + 1}`;

  const maxComparisons = Math.ceil(Math.log2(state.ranked.length + 1));
  const filmCount = Object.keys(state.films).length;
  $("progress").textContent =
    `Film ${filmCount - queue.length} of ${filmCount} · ` +
    `comparison ${comparisons + 1} of ≤${maxComparisons}`;
  $("btn-undo").classList.toggle("hidden", undoStack.length === 0);
}

function comparisonPoster(url) {
  if (!url) return "";
  try {
    const image = new URL(url);
    if (image.hostname === "a.ltrbxd.com" && image.pathname.startsWith("/resized/")) {
      image.pathname = image.pathname.replace(/-0-\d+-0-\d+-crop(?=\.)/, "-0-600-0-900-crop");
      return image.href;
    }
  } catch {}
  return url;
}

function snapshot() {
  undoStack.push({
    ranked: [...state.ranked],
    skipped: [...state.skipped],
    queue: [...queue],
    candidate,
    lo,
    hi,
    comparisons,
    placedThisSession,
  });
  if (undoStack.length > 50) undoStack.shift();
}

function undo() {
  const prev = undoStack.pop();
  if (!prev) return;
  state.ranked = prev.ranked;
  state.skipped = prev.skipped;
  queue = prev.queue;
  candidate = prev.candidate;
  lo = prev.lo;
  hi = prev.hi;
  comparisons = prev.comparisons;
  placedThisSession = prev.placedThisSession;
  saveState();
  renderComparison();
}

function place(index) {
  state.ranked.splice(index, 0, candidate);
  placedThisSession++;
  saveState();
  nextCandidate();
}

function answer(result) {
  if (!candidate) return;
  snapshot();
  const mid = Math.floor((lo + hi) / 2);
  if (result === "tie") {
    place(mid);
    return;
  }
  if (result === "better") hi = mid;
  else lo = mid + 1;
  comparisons++;
  if (lo >= hi) place(lo);
  else renderComparison();
}

function skipFilm() {
  if (!candidate) return;
  snapshot();
  state.skipped.push(candidate);
  placedThisSession++;
  saveState();
  nextCandidate();
}

function score(index, total) {
  if (total === 1) return "10.0";
  return (10 - (9 * index) / (total - 1)).toFixed(1);
}

function scoreClass(value) {
  if (value >= 7) return "hi";
  if (value >= 4) return "mid";
  return "lo";
}

function renderList() {
  const ol = $("ranking");
  ol.innerHTML = "";
  const total = state.ranked.length;
  state.ranked.forEach((key, i) => {
    const film = state.films[key];
    if (!film) return;
    const s = score(i, total);
    const li = document.createElement("li");
    li.className = "rank-row";
    li.innerHTML = `
      <span class="rank-num">${i + 1}</span>
      <img src="${film.poster || ""}" alt="" loading="lazy" data-key="${film.key}">
      <span class="rank-info">
        <a href="${film.url}" target="_blank" rel="noopener">${film.title} <span class="film-year">${film.year}</span></a>
        <span class="rank-sub">${film.rating ? `★ ${film.rating} on letterboxd` : "unrated"}${film.watched ? ` · watched ${film.watched}` : ""}</span>
      </span>
      <span class="score ${scoreClass(parseFloat(s))}">${s}</span>
      <button class="link-btn" data-rerank="${key}">re-rank</button>`;
    ol.appendChild(li);
    queuePoster(film, li.querySelector("img"), false);
  });

  const skippedUl = $("skipped");
  skippedUl.innerHTML = "";
  $("skipped-section").classList.toggle("hidden", state.skipped.length === 0);
  state.skipped.forEach((key) => {
    const film = state.films[key];
    if (!film) return;
    const li = document.createElement("li");
    li.className = "skip-row";
    li.innerHTML = `<span>${film.title} (${film.year})</span>
      <button class="link-btn" data-rerank="${key}">rank it</button>`;
    skippedUl.appendChild(li);
  });
}

function rerank(key) {
  state.ranked = state.ranked.filter((k) => k !== key);
  state.skipped = state.skipped.filter((k) => k !== key);
  queue.unshift(key);
  totalToPlace = placedThisSession + queue.length;
  saveState();
  nextCandidate();
}

$("setup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("username-input").value.trim().replace(/^@/, "");
  if (!name) return;
  $("setup-error").classList.add("hidden");
  try {
    await init(name);
  } catch (err) {
    $("setup-error").textContent = err.message;
    $("setup-error").classList.remove("hidden");
  }
});

$("side-candidate").addEventListener("click", () => answer("better"));
$("side-pivot").addEventListener("click", () => answer("worse"));
$("btn-cantrate").addEventListener("click", () => answer("tie"));
$("btn-skip").addEventListener("click", skipFilm);
$("btn-undo").addEventListener("click", undo);

$("nav-rank").addEventListener("click", () => {
  if (candidate) renderComparison();
  else if (queue.length) nextCandidate();
  else show("view-done");
});
$("nav-list").addEventListener("click", () => {
  renderList();
  show("view-list");
});
$("switch-user").addEventListener("click", () => {
  localStorage.removeItem("lbx-username");
  candidate = null;
  $("nav").classList.add("hidden");
  show("view-setup");
});

for (const id of ["btn-refresh", "btn-refresh-2"]) {
  $(id).addEventListener("click", () => init(username));
}
for (const id of ["btn-import", "btn-import-2"]) {
  $(id).addEventListener("click", importHistory);
}

document.addEventListener("click", (e) => {
  const key = e.target.dataset && e.target.dataset.rerank;
  if (key) rerank(key);
});

document.addEventListener("keydown", (e) => {
  if ($("view-compare").classList.contains("hidden")) return;
  if (e.key === "ArrowLeft" || e.key === "b") answer("better");
  else if (e.key === "ArrowRight" || e.key === "w") answer("worse");
  else if (e.key === "t") answer("tie");
  else if (e.key === "s") skipFilm();
  else if (e.key === "u") undo();
});

const savedName = localStorage.getItem("lbx-username");
$("username-input").value = savedName || "m10zeus";
if (savedName) init(savedName).catch(() => show("view-setup"));
