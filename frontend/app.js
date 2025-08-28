const API_BASE = "http://localhost:8000";
let LAST_RESULTS = [];  // cache for student details

function sanitize(s) {
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function aiBadge(v) {
  const p = +v || 0;
  const cls = p < 0.25 ? "low" : (p < 0.6 ? "mid" : "high");
  return `<span class="badge ${cls}">${Math.round(p*100)}%</span>`;
}
function setLoading(isLoading) {
  const btn = document.getElementById("run");
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "Running…" : "Run Evaluation";
}

function buildStudentMap(results) {
  const map = {};
  for (const r of results) {
    map[String(r.student_id)] = r;
  }
  return map;
}

function renderStudentDetails(rec, withFb) {
  const box = document.getElementById("details");
  if (!rec) { box.innerHTML = ""; return; }
  const tasks = (rec.results || []).slice().sort((a,b)=>{
    const na = parseInt(String(a.task||"").match(/\d+/)?.[0] || 1e9,10);
    const nb = parseInt(String(b.task||"").match(/\d+/)?.[0] || 1e9,10);
    return na-nb;
  });
  let html = `<div class="card"><h3>Student ${sanitize(rec.student_id)}</h3>
    <div><b>Final score:</b> ${sanitize(rec.final_score ?? 0)} / 100 &nbsp; | &nbsp; <b>AI:</b> ${aiBadge(rec.ai_score ?? 0)}</div>
    <div class="tasks">`;
  for (const t of tasks) {
    html += `<div class="task">
      <div class="task-head"><b>${sanitize(t.task || "task")}</b> — Score: ${sanitize(t.score ?? 0)}</div>`;
    if (withFb) {
      html += `<div class="task-fb"><b>Feedback:</b><pre>${sanitize(t.feedback || "")}</pre></div>`;
    }
    html += `</div>`;
  }
  html += `</div></div>`;
  box.innerHTML = html;
}

async function run() {
  const a = document.getElementById("assignment").files[0];
  const z = document.getElementById("subs").files[0];
  const model = document.getElementById("model").value;
  const temp = document.getElementById("temp").value;
  const showFb = document.getElementById("fb").checked;
  const batch = document.getElementById("batch").checked;

  const errBox = document.getElementById("error");
  errBox.style.display = "none"; errBox.innerHTML = "";

  if (!a || !z) {
    errBox.style.display = "block";
    errBox.innerHTML = "<b>Missing files:</b> Please choose assignment and submissions ZIP.";
    return;
  }

  setLoading(true);
  try {
    const fd = new FormData();
    fd.append("assignment", a);
    fd.append("submissions", z);
    fd.append("model", model);
    fd.append("temperature", temp);
    fd.append("show_feedback", showFb ? "true" : "false");
    fd.append("batch_per_student", batch ? "true" : "false");

    const res = await fetch(`${API_BASE}/run`, { method: "POST", body: fd });
    if (!res.ok) {
      const txt = await res.text();
      errBox.style.display = "block";
      errBox.innerHTML = `<b>Server error:</b><br/><pre>${sanitize(txt)}</pre>`;
      return;
    }
    const data = await res.json();
    LAST_RESULTS = data.results || [];

    // summary
    const s = data.summary || {class_avg:0, ai_avg:0};
    document.getElementById("summary").innerHTML =
      `<b>Class avg:</b> ${Number(s.class_avg).toFixed(2)} &nbsp; | &nbsp; <b>AI avg:</b> ${Number(s.ai_avg).toFixed(3)}`;

    // downloads via server URLs
    const csvLink = document.getElementById("dl-csv");
    const jsonlLink = document.getElementById("dl-jsonl");
    csvLink.href = `${API_BASE}${data.download.csv}`;
    jsonlLink.href = `${API_BASE}${data.download.jsonl}`;

    // build task header list
    const taskSet = new Set();
    for (const r of LAST_RESULTS) {
      for (const t of (r.results || [])) {
        const m = String(t.task || "").match(/(\d+)/);
        const tid = m ? `task${m[1]}` : String(t.task||"task1");
        taskSet.add(tid);
      }
    }
    const tasks = Array.from(taskSet).sort((a,b)=>{
      const na = parseInt(a.match(/\d+/)?.[0] || 1e9,10);
      const nb = parseInt(b.match(/\d+/)?.[0] || 1e9,10);
      return na-nb;
    });

    // header
    let thead = "<tr><th>student_id</th><th>final_score</th><th>ai_score</th>";
    for (const t of tasks) {
      thead += `<th>${t}_score</th>`;
      if (showFb) thead += `<th>${t}_feedback</th>`;
    }
    thead += "</tr>";

    // rows
    let rows = "";
    for (const r of LAST_RESULTS) {
      const sid = sanitize(r.student_id);
      const finalScore = Number(r.final_score ?? 0).toFixed(0);
      const ai = Number(r.ai_score ?? 0);
      const map = {};
      for (const t of (r.results || [])) {
        const m = String(t.task || "").match(/(\d+)/);
        const tid = m ? `task${m[1]}` : String(t.task||"task1");
        map[tid] = t;
      }
      let cells = `<td class="student-cell" data-sid="${sid}">${sid}</td><td>${finalScore}</td><td>${aiBadge(ai)}</td>`;
      for (const t of tasks) {
        const it = map[t] || {};
        cells += `<td>${sanitize(it.score ?? 0)}</td>`;
        if (showFb) cells += `<td>${sanitize(it.feedback ?? "")}</td>`;
      }
      rows += `<tr>${cells}</tr>`;
    }

    document.getElementById("table").innerHTML =
      `<div class="card"><table><thead>${thead}</thead><tbody>${rows}</tbody></table></div>
       <div id="details"></div>`;

    // attach click handlers for student details
    const studentCells = document.querySelectorAll(".student-cell");
    const stuMap = buildStudentMap(LAST_RESULTS);
    studentCells.forEach(td => {
      td.addEventListener("click", () => {
        const sid = td.getAttribute("data-sid");
        renderStudentDetails(stuMap[sid], showFb);
        td.scrollIntoView({behavior:"smooth", block:"center"});
      });
    });

  } catch (e) {
    errBox.style.display = "block";
    errBox.innerHTML = `<b>Error:</b><br/><pre>${sanitize(String(e))}</pre>`;
  } finally {
    setLoading(false);
  }
}

document.getElementById("run").addEventListener("click", run);

