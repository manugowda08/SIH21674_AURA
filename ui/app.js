// SIH26174 monitoring dashboard.
// Polls /state rather than using websockets: one fewer moving part, and a
// dropped poll self-heals on the next tick instead of needing a reconnect.

var POLL_MS = 400;
var MARK = { done: "\u2713", current: "\u203a", next: "", pending: "", skipped: "!" };
var missed = 0;

function esc(s) {
  return String(s).replace(/[&<>]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
  });
}

function el(id) { return document.getElementById(id); }

function renderSteps(steps) {
  return steps.map(function (s) {
    return '<div class="step ' + s.state + '">' +
           '<span class="mark">' + (MARK[s.state] || "") + "</span>" +
           '<span class="nm">' + s.id + ". " + esc(s.name) + "</span></div>";
  }).join("");
}

function pretty(name) {
  return String(name || "").replace(/_/g, " ");
}

function renderAlerts(alerts) {
  if (!alerts.length) return '<div class="quiet">none</div>';
  return alerts.slice(-5).map(function (a) {
    // Tolerate a plain string too, so an older backend cannot blank the panel.
    var msg = (typeof a === "string") ? a : a.message;
    var why = "";
    if (a && a.expected) {
      why = '<span class="exp">expected: ' + esc(pretty(a.expected)) +
            "  \u2192  detected: " + esc(pretty(a.detected || "idle")) + "</span>";
    }
    return '<div class="alert">' + esc(msg) + why + "</div>";
  }).join("");
}

// One bar per class, the winner highlighted. Showing the whole distribution,
// not only the top label, is what lets a viewer see the model hesitate:
// a peaked distribution is a decision, a flat one is a guess.
function renderProbs(probs) {
  if (!probs || !probs.length) return '<div class="quiet">waiting for data</div>';
  var top = 0;
  probs.forEach(function (p, i) { if (p.p > probs[top].p) top = i; });
  return probs.map(function (p, i) {
    var pct = Math.round(p.p * 100);
    return '<div class="prow' + (i === top ? " top" : "") + '">' +
           '<span class="pn">' + esc(pretty(p.name)) + "</span>" +
           '<span class="ptrack"><i class="pfill" style="width:' + pct + '%"></i></span>' +
           '<span class="pval">' + pct + "%</span></div>";
  }).join("");
}

function renderLog(entries) {
  return entries.slice(-40).reverse().map(function (e) {
    return "<b>" + e.t.toFixed(1) + "s</b>  " + esc(e.event) +
           (e.step ? "  step " + e.step : "");
  }).join("<br>");
}

function apply(s) {
  missed = 0;
  el("dot").className = "dot";
  el("proto").textContent = s.protocol || "";
  el("fps").textContent = (s.fps || 0).toFixed(1) + " fps";
  el("rec").className = s.recording ? "tag on" : "tag";

  var cur = el("cur");
  cur.textContent = s.current || "idle";
  cur.className = s.current ? "big" : "big idle";
  el("bar").style.width = ((s.confidence || 0) * 100) + "%";

  el("next").textContent = s.next_prompt || "protocol complete";
  el("steps").innerHTML = renderSteps(s.steps || []);
  el("alerts").innerHTML = renderAlerts(s.alerts || []);
  el("probs").innerHTML = renderProbs(s.probs || []);
  el("log").innerHTML = renderLog(s.log || []);
  document.body.classList.toggle("alerting", !!s.alerting);
  if (s.alerting && s.alerts.length) {
    var last = s.alerts[s.alerts.length - 1];
    var txt = (typeof last === "string") ? last : last.message;
    if (last && last.expected) {
      txt += "   (expected " + pretty(last.expected) + ", detected " +
             pretty(last.detected || "idle") + ")";
    }
    el("alerttext").textContent = txt;
  }

  el("reportnote").textContent = s.report_note || "";
  el("sess").textContent = "session " + (s.session || "--");
  el("events").textContent = (s.n_events || 0) + " events";
}

function tick() {
  fetch("/state")
    .then(function (r) { return r.json(); })
    .then(apply)
    .catch(function () {
      // Tolerate a couple of dropped polls before flagging the backend as gone;
      // a single timeout during heavy inference is normal, not a failure.
      if (++missed > 3) el("dot").className = "dot stale";
    });
}

setInterval(tick, POLL_MS);
tick();


// ---------------------------------------------------------------- voice
// The voice list is discovered by the audio thread after the dashboard is
// already up, so the first request can legitimately come back empty. Retry
// briefly rather than showing a permanently dead control.
var voices = [], voiceTries = 0;

function firstOf(gender) {
  for (var i = 0; i < voices.length; i++) {
    if (voices[i].gender === gender) return voices[i].i;
  }
  return -1;
}

function paintVoice(current) {
  var sel = el("voicesel");
  sel.value = String(current);
  var cur = voices.filter(function (v) { return v.i === current; })[0];
  el("vfemale").classList.toggle("active", !!cur && cur.gender === "Female");
  el("vmale").classList.toggle("active", !!cur && cur.gender === "Male");
}

function chooseVoice(index) {
  if (index < 0) return;
  fetch("/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: index })
  }).then(function (r) { return r.json(); })
    .then(function (d) { if (d.ok) paintVoice(index); })
    .catch(function () {});
}

function loadVoices() {
  fetch("/voices").then(function (r) { return r.json(); }).then(function (d) {
    if (!d.voices || !d.voices.length) {
      if (++voiceTries < 15) { setTimeout(loadVoices, 1000); return; }
      el("voicesel").innerHTML = "<option>no voices available</option>";
      return;
    }
    voices = d.voices;
    var sel = el("voicesel");
    sel.innerHTML = voices.map(function (v) {
      return '<option value="' + v.i + '">' + esc(v.label) +
             (v.gender ? " (" + v.gender + ")" : "") + "</option>";
    }).join("");
    sel.disabled = false;
    el("vfemale").disabled = firstOf("Female") < 0;
    el("vmale").disabled = firstOf("Male") < 0;
    paintVoice(d.current);
  }).catch(function () {
    if (++voiceTries < 15) setTimeout(loadVoices, 1000);
  });
}

el("voicesel").addEventListener("change", function () {
  chooseVoice(parseInt(this.value, 10));
});
el("vfemale").addEventListener("click", function () { chooseVoice(firstOf("Female")); });
el("vmale").addEventListener("click", function () { chooseVoice(firstOf("Male")); });
loadVoices();