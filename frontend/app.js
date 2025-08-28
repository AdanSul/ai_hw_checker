const API_BASE = "http://localhost:8000";
let LAST_RESULTS = [];
let TABLE_READY_HTML = ""; // cache rendered table HTML

// utils
function sanitize(s){ return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function aiBadge(v){ const p=+v||0; const cls=p<0.25?"low":(p<0.6?"mid":"high"); return `<span class="badge ${cls}">${Math.round(p*100)}%</span>`; }
function setLoading(on){ const btn=document.getElementById("run"); btn.disabled=on; btn.textContent=on?"Running…":"Run Evaluation"; }
function showStatus(msg){ const s=document.getElementById("status"); s.classList.add("show"); s.innerHTML=msg; }
function hideStatus(){ const s=document.getElementById("status"); s.classList.remove("show"); s.innerHTML=""; }
function showError(html){ const e=document.getElementById("error"); e.style.display="block"; e.innerHTML=html; }
function clearError(){ const e=document.getElementById("error"); e.style.display="none"; e.innerHTML=""; }
function setDownload(el, urlOrNull, titleWhenDisabled){
  if(!el) return;
  if(urlOrNull){ el.classList.remove("disabled"); el.removeAttribute("aria-disabled"); el.removeAttribute("title"); el.href=urlOrNull; }
  else{ el.classList.add("disabled"); el.setAttribute("aria-disabled","true"); el.setAttribute("title", titleWhenDisabled||"Not available"); el.removeAttribute("href"); }
}
function disableDownloads(){
  setDownload(document.getElementById("dl-csv"), null, "No CSV yet");
  setDownload(document.getElementById("dl-jsonl"), null, "No JSONL yet");
}
function wireFileChips(){
  const setTag=(id,file)=>{ const el=document.getElementById(id); if(!el) return;
    if(file){ el.textContent=`✓ ${file.name}`; el.classList.add("ok"); } else { el.textContent="Drag & drop to upload"; el.classList.remove("ok"); }
  };
  document.getElementById("assignment")?.addEventListener("change", e => { setTag("assignment-tag", e.target.files[0]); clearError(); });
  document.getElementById("subs")?.addEventListener("change", e => { setTag("subs-tag", e.target.files[0]); clearError(); });
}
function buildStudentMap(results){ const m={}; for(const r of results) m[String(r.student_id)]=r; return m; }
function renderStudentDetails(rec, withFb){
  const box=document.getElementById("details");
  if(!rec){ box.style.display="none"; box.innerHTML=""; return; }
  const tasks=(rec.results||[]).slice().sort((a,b)=>{
    const na=parseInt(String(a.task||"").match(/\d+/)?.[0]||1e9,10);
    const nb=parseInt(String(b.task||"").match(/\d+/)?.[0]||1e9,10);
    return na-nb;
  });
  let html=`<h3 style="margin-top:0">Student ${sanitize(rec.student_id)}</h3>
    <div style="margin-bottom:8px"><b>Final score:</b> ${sanitize(rec.final_score ?? 0)} / 100 &nbsp; | &nbsp; <b>AI:</b> ${aiBadge(rec.ai_score ?? 0)}</div>
    <div class="tasks">`;
  for(const t of tasks){
    html+=`<div class="task">
      <div class="task-head"><b>${sanitize(t.task || "task")}</b> — Score: ${sanitize(t.score ?? 0)}</div>`;
    if(withFb) html+=`<div class="task-fb"><b>Feedback:</b><pre>${sanitize(t.feedback || "")}</pre></div>`;
    html+=`</div>`;
  }
  html+=`</div>`;
  box.innerHTML=html; box.style.display="block";
}

// render (table) -> returns HTML string
function renderTableHTML(results, showFb){
  const taskSet=new Set();
  for(const r of results){
    for(const t of (r.results||[])){
      const m=String(t.task||"").match(/(\d+)/); const tid=m?`task${m[1]}`:String(t.task||"task1"); taskSet.add(tid);
    }
  }
  const tasks=Array.from(taskSet).sort((a,b)=>{
    const na=parseInt(a.match(/\d+/)?.[0]||1e9,10); const nb=parseInt(b.match(/\d+/)?.[0]||1e9,10); return na-nb;
  });

  let thead="<tr><th>student_id</th><th>final_score</th><th>ai_score</th>";
  for(const t of tasks){
    thead+=`<th>${t}_score</th>`;
    if(showFb) thead+=`<th>${t}_feedback</th>`;
  }
  thead+="</tr>";

  let rows="";
  for(const r of results){
    const sid=sanitize(r.student_id);
    const finalScore=Number(r.final_score ?? 0).toFixed(0);
    const ai=Number(r.ai_score ?? 0);

    const map={};
    for(const t of (r.results||[])){
      const m=String(t.task||"").match(/(\d+)/); const tid=m?`task${m[1]}`:String(t.task||"task1"); map[tid]=t;
    }
    let cells=`<td class="student-cell" data-sid="${sid}">${sid}</td><td>${finalScore}</td><td>${aiBadge(ai)}</td>`;
    for(const t of tasks){
      const it=map[t]||{};
      cells+=`<td>${sanitize(it.score ?? 0)}</td>`;
      if(showFb){
        const txt=sanitize(it.feedback || "");
        const needsMore = (it.feedback||"").length > 220;
        const fbCell = needsMore
          ? `<div class="feedback clamp">${txt}</div><span class="cell-more">more</span>`
          : `<div class="feedback">${txt}</div>`;
        cells+=`<td>${fbCell}</td>`;
      }
    }
    rows+=`<tr>${cells}</tr>`;
  }
  return `<div class="card"><table><thead>${thead}</thead><tbody>${rows}</tbody></table></div>`;
}

async function run(){
  clearError(); hideStatus(); showStatus("Preparing files…");

  const a   = document.getElementById("assignment")?.files?.[0];
  const z   = document.getElementById("subs")?.files?.[0];
  const mdl = document.getElementById("model-select")?.value ?? "gpt-4o-mini";
  const tmp = document.getElementById("temp-number")?.value ?? 0.1;
  const fb  = document.getElementById("fb")?.checked ?? true;
  const bat = document.getElementById("batch")?.checked ?? false;

  if(!a || !z){ showError("<b>Missing files:</b> Please choose both Assignment (md/txt) and Submissions (zip)."); return; }

  setLoading(true); disableDownloads(); showStatus("Uploading & running…");
  try{
    const fd=new FormData();
    fd.append("assignment", a); fd.append("submissions", z);
    fd.append("model", mdl); fd.append("temperature", tmp);
    fd.append("show_feedback", fb ? "true" : "false");
    fd.append("batch_per_student", bat ? "true" : "false");

    const res=await fetch(`${API_BASE}/run`, { method:"POST", body:fd });
    if(!res.ok){ const txt=await res.text(); showError(`<b>Server error:</b><br/><pre>${sanitize(txt)}</pre>`); return; }
    const data=await res.json();
    LAST_RESULTS = data.results || [];

    // summary
    const s=data.summary || {class_avg:0, ai_avg:0};
    document.getElementById("summary").innerHTML =
      `<b>Class avg:</b> ${Number(s.class_avg).toFixed(2)} &nbsp; | &nbsp; <b>AI avg:</b> ${Number(s.ai_avg).toFixed(3)}`;

    // downloads
    const csvUrl   = data?.download?.csv   ? `${API_BASE}${data.download.csv}`   : null;
    const jsonlUrl = data?.download?.jsonl ? `${API_BASE}${data.download.jsonl}` : null;
    setDownload(document.getElementById("dl-csv"), csvUrl, "No CSV yet");
    setDownload(document.getElementById("dl-jsonl"), jsonlUrl, "No JSONL available");

    // render table HTML but אל תציגי עדיין
    TABLE_READY_HTML = renderTableHTML(LAST_RESULTS, fb);
    const tableEl = document.getElementById("table");
    tableEl.innerHTML = TABLE_READY_HTML;

    // הפעלת toggler
    const wrap = document.getElementById("table-wrap");
    const tog  = document.getElementById("toggle-results");
    tog.classList.remove("disabled"); tog.removeAttribute("aria-disabled"); tog.removeAttribute("title");
    tog.onclick = () => {
      const hidden = wrap.classList.toggle("hidden");
      tog.textContent = hidden ? "Show results" : "Hide results";
    };
    // מראה כברירת מחדל אחרי ריצה
    wrap.classList.remove("hidden");
    tog.textContent = "Hide results";

    // more/less באלמנטי פידבק
    document.querySelectorAll(".cell-more").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        const fbNode = btn.previousElementSibling;
        fbNode.classList.toggle("expand");
        fbNode.classList.toggle("clamp");
        btn.textContent = fbNode.classList.contains("expand") ? "less" : "more";
      });
    });

    // student details on click
    const stuMap = buildStudentMap(LAST_RESULTS);
    document.querySelectorAll(".student-cell").forEach(td=>{
      td.addEventListener("click", ()=>{
        const sid = td.getAttribute("data-sid");
        renderStudentDetails(stuMap[sid], fb);
        document.getElementById("details").scrollIntoView({behavior:"smooth", block:"nearest"});
      });
    });

    hideStatus();
  }catch(e){
    showError(`<b>Error:</b><br/><pre>${sanitize(String(e))}</pre>`);
  }finally{
    setLoading(false);
  }
}

// init
document.addEventListener("DOMContentLoaded", ()=>{
  document.getElementById("run")?.addEventListener("click", run);
  wireFileChips();
  disableDownloads();
});
